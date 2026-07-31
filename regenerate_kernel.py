#!/usr/bin/env python3
"""
Regenera o kernel Jupyter com CUDA para o ambiente atual do devbox.
Execute dentro do devbox shell.
"""

import json
import os
import sys
import glob

def main():
    python_exe = sys.executable
    
    ld_paths = []
    
    for p in ['/run/opengl-driver/lib', '/run/opengl-driver-32/lib']:
        if os.path.exists(p) and p not in ld_paths:
            ld_paths.append(p)
    
    current_ld = os.environ.get('LD_LIBRARY_PATH', '')
    for p in current_ld.split(':'):
        if p and os.path.exists(p) and '.venv' not in p and not p.startswith('/usr/') and p != '/lib64':
            if p not in ld_paths:
                ld_paths.append(p)
    
    for pattern in ['/nix/store/*-gcc-*/lib']:
        for p in glob.glob(pattern):
            if p not in ld_paths:
                ld_paths.append(p)
    
    ld_library_path = ':'.join(ld_paths)
    
    env_vars = {
        'LD_LIBRARY_PATH': ld_library_path,
        'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', '0'),
        '__NV_PRIME_RENDER_OFFLOAD': os.environ.get('__NV_PRIME_RENDER_OFFLOAD', '1'),
        '__GLX_VENDOR_LIBRARY_NAME': os.environ.get('__GLX_VENDOR_LIBRARY_NAME', 'nvidia'),
        'PATH': os.environ.get('PATH', ''),
    }
    
    wrapper_dir = os.path.expanduser("~/.local/bin")
    os.makedirs(wrapper_dir, exist_ok=True)
    wrapper_path = os.path.join(wrapper_dir, "python-cuda-devbox")
    
    wrapper = f"""#!/usr/bin/env bash
unset LD_LIBRARY_PATH
unset NIX_LD_LIBRARY_PATH
export LD_LIBRARY_PATH="{ld_library_path}"
export CUDA_VISIBLE_DEVICES="{env_vars['CUDA_VISIBLE_DEVICES']}"
export __NV_PRIME_RENDER_OFFLOAD="{env_vars['__NV_PRIME_RENDER_OFFLOAD']}"
export __GLX_VENDOR_LIBRARY_NAME="{env_vars['__GLX_VENDOR_LIBRARY_NAME']}"
export PATH="{env_vars['PATH']}"
exec {python_exe} "$@"
"""
    
    with open(wrapper_path, 'w') as f:
        f.write(wrapper)
    os.chmod(wrapper_path, 0o755)
    
    kernel = {
        "argv": [wrapper_path, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Python (Devbox CUDA)",
        "language": "python",
        "env": env_vars,
        "metadata": {"debugger": True}
    }
    
    kernel_dir = os.path.expanduser("~/.local/share/jupyter/kernels/devbox-cuda")
    os.makedirs(kernel_dir, exist_ok=True)
    
    with open(os.path.join(kernel_dir, "kernel.json"), 'w') as f:
        json.dump(kernel, f, indent=2)
    
    print(f"✅ Kernel atualizado: {kernel_dir}/kernel.json")

if __name__ == '__main__':
    main()
