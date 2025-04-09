from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import sys
import os
import subprocess

# Determine if we're on a RISC-V system
def is_riscv():
    try:
        machine = subprocess.check_output(['uname', '-m']).decode('utf-8').strip()
        return 'riscv' in machine
    except:
        return False

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_call(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the extension")
        
        for ext in self.extensions:
            self.build_extension(ext)
    
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
            '-DPYTHON_EXECUTABLE=' + sys.executable
        ]
        
        # Add RVV-specific flags if on RISC-V
        if is_riscv():
            cmake_args.append('-DUSE_RVV=ON')
        else:
            cmake_args.append('-DUSE_RVV=OFF')
        
        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]
        
        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
        build_args += ['--', '-j2']
        
        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version()
        )
        
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
            
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, 
                              cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, 
                              cwd=self.build_temp)

setup(
    name="rvv_simd",
    version="0.1.0",
    author="RISC-V Vector Library Team",
    author_email="your.email@example.com",
    description="Python bindings for RVV-SIMD: RISC-V Vector SIMD Library",
    long_description=open("../README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/rvv-simd",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.6",
    ext_modules=[CMakeExtension("rvv_simd.rvv_simd")],
    cmdclass={"build_ext": CMakeBuild},
    install_requires=[
        "numpy>=1.16.0",
    ],
    extras_require={
        "test": ["pytest>=4.6.0"],
        "benchmark": ["matplotlib>=3.0.0", "pandas>=0.24.0"],
    },
)
