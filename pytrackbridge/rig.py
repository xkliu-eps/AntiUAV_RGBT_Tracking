import re
import sys
import os
import runpy
import importlib.util
import importlib
import pickle
from importlib.machinery import SourceFileLoader
import urllib.request
from urllib.error import URLError

# Global config object, dynamically loaded from a config file at runtime.
cfg = None


class RigError(Exception):
    """Custom exception for recoverable config / replacement errors."""
    pass


def apply_replacement(module_name, func_path, new_func_name, show=True):
    """
    Apply a single runtime function replacement (monkey-patch).

    Workflow:
    1. Ensure the target module is loaded into sys.modules.
    2. Locate the replacement function (first from the cfg object, then
       fall back to importing from an external module).
    3. Walk the dotted attribute path to reach the owning object.
    4. Save the original function as <func_name>_old, then install the
       new function via setattr.

    Args:
        module_name:   Target module, e.g. 'pytracking.evaluation.tracker'.
        func_path:     Dotted attribute path to the target function, e.g.
                       'Tracker._read_image' (supports nesting like 'obj.sub.func').
        new_func_name: Name of the replacement function. Looked up on cfg first;
                       if not found and contains '.', imported from another module.
        show:          Whether to print replacement info (True in direct mode,
                       False in callback mode).

    Raises:
        RigError: If the module cannot be imported, the replacement function
                  cannot be found, or the attribute path does not exist.
    """
    # An empty replacement name means no-op (documentation-only entry).
    if not new_func_name:
        return

    # ── 1. Ensure the target module is loaded into sys.modules ──
    if module_name not in sys.modules:
        try:
            importlib.import_module(module_name)
        except ImportError as e:
            raise RigError(f"Failed to import module {module_name}: {e}")
    module = sys.modules[module_name]

    # ── 2. Locate the replacement function ──
    # First try to get it as an attribute of the cfg object.
    new_func = getattr(cfg, new_func_name, None)

    # If not found on cfg and the name contains '.', treat it as
    # "module.attribute" and import from an external module.
    if not callable(new_func) and '.' in new_func_name:
        mod_name, attr_name = new_func_name.rsplit('.', 1)
        try:
            new_func = getattr(importlib.import_module(mod_name), attr_name)
        except (ImportError, AttributeError) as e:
            raise RigError(
                f"Failed to import {attr_name} from {mod_name}: {e}"
            )

    if not callable(new_func):
        raise RigError(
            f"Replacement function not found: {new_func_name}"
        )

    # ── 3. Walk the dotted attribute path (supports nested classes/objects) ──
    # e.g. func_path = 'Tracker._read_image':
    #   parts = ['Tracker', '_read_image']
    #   Navigate: module -> module.Tracker, then patch _read_image on Tracker.
    parts = func_path.split('.')
    for p in parts[:-1]:
        if not hasattr(module, p):
            raise RigError(
                f"Attribute path not found in module {module_name}: "
                f"{'.'.join(parts[:parts.index(p) + 1])}"
            )
        module = getattr(module, p)  # Step into nested attribute.
    func_name = parts[-1]  # The actual function name to replace.

    if show:
        print(f"{module_name} @ {func_path} ==> {new_func_name}")

    # ── 4. Save original function and install the replacement ──
    # The original is saved as <func_name>_old so it can be restored or
    # chain-called later.
    if not hasattr(module, func_name):
        raise RigError(
            f"Target function '{func_name}' does not exist on the module/object; "
            f"cannot replace."
        )
    setattr(module, func_name + "_old", getattr(module, func_name))
    setattr(module, func_name, new_func)


def parse_replacements(text):
    """
    Parse a flat list of replacement rules from text.

    Each line format:
        module @ func_old -> new_func  # optional comment
    Or (no replacement target, documentation-only):
        module @ func_old

    Example:
        pytracking.evaluation.tracker @ Tracker._read_image -> read_image

    Returns:
        list of (module, func_old, func_new, comment)

    Raises:
        RigError: If a line is malformed (missing ' @ ').
    """
    replacements = []

    for line in text.splitlines():
        # Separate the comment portion (everything after '#').
        line_split = line.split('#', 1)
        clean_line = line_split[0].strip()
        comment = line_split[1].strip() if len(line_split) > 1 else ''

        # Skip blank lines.
        if not clean_line:
            continue

        # Every line must contain ' @ ' to separate module and function parts.
        if ' @ ' not in clean_line:
            raise RigError(f"Malformed line (missing ' @ '): {clean_line}")

        module_part, func_part = clean_line.split(' @ ', 1)
        module_part = module_part.strip()
        func_part = func_part.strip()

        # Parse '->': left side is the old function, right side the new one.
        if '->' in func_part:
            func_old, func_new = map(str.strip, func_part.split('->', 1))
        else:
            func_old = func_part
            func_new = ""

        replacements.append((module_part, func_old, func_new, comment))

    return replacements


def _merge_args(base_args, override_args):
    """
    Merge two argument lists, with override_args taking priority.

    For ``--key=value`` or ``--flag`` style args, matching keys in base_args
    are removed.  Positional arguments are kept from both lists (base first,
    override appended).
    """
    if not override_args:
        return base_args

    def _arg_key(arg):
        """Extract the key portion of an argument, e.g. '--debug' from '--debug=1'."""
        if arg.startswith('--'):
            return arg.split('=')[0]
        return arg

    override_keys = {_arg_key(a) for a in override_args if a.startswith('--')}
    filtered_base = [
        a for a in base_args
        if not a.startswith('--') or _arg_key(a) not in override_keys
    ]
    return filtered_base + override_args


def main(called=None, loadCache=False):
    """
    Main entry point.

    Two operating modes:
    ┌────────────────┬──────────────────────────────────────────────────┐
    │ Direct mode    │ called is None                                   │
    │                │ load config → parse rules → apply patches → run  │
    ├────────────────┼──────────────────────────────────────────────────┤
    │ Callback mode  │ called is a non-empty dict (invoked internally   │
    │                │ by the target script)                            │
    │                │ silent patch → unpickle args → call entry func   │
    └────────────────┴──────────────────────────────────────────────────┘

    Callback mode exists to support multi-process scenarios: child
    processes need the same monkey-patch environment but should not
    re-execute the entire target script — they only need the patches
    reapplied and then call the designated entry function.

    Args:
        called: None for direct mode; a dict for callback mode containing:
                config, run_module, run_function, run_argv.
        loadCache: When False (default), the hidden copy is rebuilt from the
                   original script (include merge + markers).  When True, the
                   existing hidden copy is loaded directly.  Child processes
                   in callback mode should pass True to avoid re-building.
    """
    if called is None:
        called = {}
    input_args = []

    # ── Determine the config file path ──
    if len(called) != 0:
        # Callback mode: the 'called' dict carries the config path.
        sys.argv = ["rig.py", called["config"]]
    else:
        if len(sys.argv) < 2:
            print("Usage: python rig.py <config.py> [extra_args...]")
            sys.exit(1)
        input_args = sys.argv[2:]  # extra args after the config path

    global cfg
    config_path = os.path.abspath(sys.argv[1])
    if not os.path.isfile(config_path):
        config_path = os.path.abspath('script/' + sys.argv[1])
        if not os.path.isfile(config_path):
            print(f"Config file not found: {config_path}")
            sys.exit(1)

    # ── Prepare the config file for loading ──
    # Two paths are used:
    #   - config_path:    the original user-written script (never modified)
    #   - injected_path:  a hidden copy where include content is merged in
    #
    # When loadCache is False (main process), we build the hidden copy:
    #   1. Read the original script
    #   2. Resolve '# include "module.path"' and read the included file
    #   3. Write a merged version back to the hidden copy:
    #      - included content is prefixed with '#<temporary code> ' so it
    #        can be visually distinguished when debugging / setting breakpoints
    #      - a split-line comment separates included content from original code
    #   4. The hidden copy is then loaded by InjectLoader
    #
    # When loadCache is True (callback mode / child processes), we skip the
    # merge step and load the already-built hidden copy directly.
    dir_name = os.path.dirname(config_path)
    base_name = os.path.basename(config_path)
    injected_path = os.path.join(dir_name, base_name)
    
    if not loadCache:
        injected_path = os.path.join(dir_name, '.' + base_name)
        with open(config_path, 'r', encoding='utf-8') as f:
            code = f.read()
        head = ''
        if m := re.search(r'#\s*include\s*"([^"]+)"', code):
            fileName = m[1].replace('.', '/') + '.py'
            with open(fileName, 'r', encoding='utf-8') as f:
                head = f.read() 

        # Prefix every line of the included content with '#<temporary code> '.
        # This makes it easy to see which code comes from the include file
        # when opening the hidden copy in an editor or debugger.
        mark_head = '\n'.join('#<temporary code> ' + l for l in head.splitlines())
        # A visible separator between included code and original script.
        mark_code = mark_head + '\n\n# ====================== split line ======================\n\n' + code
        with open(injected_path, 'w', encoding='utf-8') as f:
            f.write(mark_code)
    
    # ── Custom loader: strip temporary-code markers at runtime ──
    # The hidden copy contains '#<temporary code> ' prefixes for debugging.
    # At runtime we simply strip those prefixes so Python sees clean code.
    # The co_filename in traceback will still point to the hidden copy,
    # making breakpoints and line numbers match the physical file.
    class InjectLoader(SourceFileLoader):
        def get_data(self, path):
            if not path.endswith('.py'):
                return super().get_data(path)
            data = super().get_data(path)
            code = data.decode('utf-8')
            # Remove the debug marker so Python executes normal code.
            code = code.replace('#<temporary code> ', '')
            return code.encode('utf-8')

    spec = importlib.util.spec_from_file_location("cfg", injected_path, loader=InjectLoader("cfg", injected_path))
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)  # Execute the config; its top-level
    # variables become attributes of cfg.

    # ── Read config entries ──
    # target_script:  path to the target script to run after patching.
    # script_args:    command-line arguments passed to the target script.
    # call_tree_text: the replacement rules (one per line).
    target = getattr(cfg, "target_script", None)
    args = getattr(cfg, "script_args", [])
    tree_text = getattr(cfg, "call_tree_text", "")

    # Validate target script: an empty string means "use cfg.main instead".
    if (not target or not os.path.isfile(target)) and target != '':
        print(f"Error: target_script <{target}> is missing or invalid")
        sys.exit(1)

    # ── Parse the replacement rules ──
    try:
        replacements = parse_replacements(tree_text)
    except RigError as e:
        print(f"Failed to parse replacement rules: {e}")
        sys.exit(1)

    if len(called) == 0:
        # ============================
        # Direct mode
        # ============================
        print('*' * 40)

        # Step 1: iterate over the replacement list and apply each patch.
        try:
            for mod, func_old, func_new, _ in replacements:
                apply_replacement(mod, func_old, func_new, show=True)
        except RigError as e:
            print(f"Function replacement failed: {e}")
            sys.exit(1)

        print('*' * 40)

        # Step 2: run the target script.
        if len(target) != 0:
            # Merge config script_args with command-line extra args.
            # input_args take priority: same --key overrides the config value.
            sys.argv = [target] + _merge_args(args, input_args)
            print("command start with " + " ".join(sys.argv))
            runpy.run_path(target, run_name='__main__')
        elif getattr(cfg, "main", None) is not None:
            # If the config defines a main() function, call it directly.
            print("command start in main of " + sys.argv[1])
            return cfg.main()
        else:
            print("no command to run")

    else:
        # ============================
        # Callback mode (multi-process child entry point)
        # ============================

        # Step 1: silently apply all replacements (no output).
        try:
            for mod, func_old, func_new, _ in replacements:
                apply_replacement(mod, func_old, func_new, show=False)
        except RigError as e:
            print(f"Callback mode replacement failed: {e}")
            sys.exit(1)

        # Step 2: restore arguments from a pickle file and invoke the
        # designated entry function.
        # The 'called' dict is serialized by the parent process and contains:
        #   run_module:   module name of the entry point.
        #   run_function: function name to call.
        #   run_argv:     path to the pickle file holding the arguments.
        run_module = called["run_module"]
        run_function = called["run_function"]
        run_argv = called["run_argv"]
        
        if run_argv.startswith("http://127.0.0.1"):
            try:
                with urllib.request.urlopen(run_argv, timeout=5) as resp:
                    obj = pickle.loads(resp.read())
            except (URLError, pickle.UnpicklingError, EOFError) as e:
                print(f"Failed to fetch pickle via HTTP ({run_argv}): {e}")
                sys.exit(1)
        else:
            try:
                with open(run_argv, "rb") as f:
                    obj = pickle.load(f)
            except (FileNotFoundError, pickle.UnpicklingError, EOFError) as e:
                print(f"Failed to read pickle args file ({run_argv}): {e}")
                sys.exit(1)

        importlib.import_module(run_module)
        run_module_ref = sys.modules[run_module]
        getattr(run_module_ref, run_function)(*obj)


if __name__ == '__main__':
    main()
