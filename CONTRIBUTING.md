# Contributing to RVV-SIMD

Thank you for your interest in contributing to the RVV-SIMD library! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Review Process](#review-process)

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please read and understand the [Code of Conduct](CODE_OF_CONDUCT.md) before contributing.

## Getting Started

### Prerequisites

- RISC-V toolchain with vector extension support
- CMake (3.10+)
- Python 3.6+ (for Python bindings)
- pybind11 (for Python bindings)
- Google Test (for testing)
- Google Benchmark (for benchmarking)

### Setting Up Development Environment

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/rvv-simd.git
   cd rvv-simd
   ```

3. Add the original repository as an upstream remote:
   ```bash
   git remote add upstream https://github.com/originalowner/rvv-simd.git
   ```

4. Create a build directory and build the project:
   ```bash
   mkdir build && cd build
   cmake ..
   make
   ```

5. Run the tests to ensure everything is working:
   ```bash
   make test
   ```

## Development Workflow

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes, following the coding standards

3. Add tests for your changes

4. Update documentation as necessary

5. Run the tests to ensure they pass:
   ```bash
   cd build
   make test
   ```

6. Run the benchmarks to ensure performance is maintained or improved:
   ```bash
   cd build
   make rvv_simd_benchmarks
   ./benchmarks/rvv_simd_benchmarks
   ```

7. Commit your changes with a descriptive commit message:
   ```bash
   git commit -m "Add feature: your feature description"
   ```

8. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

9. Create a pull request from your fork to the main repository

## Coding Standards

### C++ Code

- Follow the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) with the following exceptions:
  - Use `snake_case` for function and variable names
  - Use `CamelCase` for class and struct names
  - Use `UPPER_CASE` for constants and macros

- Use modern C++ features (C++14 or later)

- Include appropriate comments and documentation

- Optimize for readability and maintainability

### Python Code

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide

- Use type hints where appropriate

- Include docstrings for all functions, classes, and modules

### Commit Messages

- Use the imperative mood ("Add feature" not "Added feature")
- Start with a capital letter
- Keep the first line under 50 characters
- Provide a detailed description in the body if necessary
- Reference issues and pull requests where appropriate

## Testing

- All new features should include appropriate tests
- All bug fixes should include tests that reproduce the bug
- Tests should be written using Google Test
- Tests should be fast and deterministic
- Tests should not depend on external resources

## Documentation

- Update documentation for any new features or changes to existing features
- Documentation should be clear, concise, and accurate
- Include examples where appropriate
- Update the README.md file if necessary

## Submitting Changes

1. Ensure all tests pass
2. Ensure the code follows the coding standards
3. Ensure documentation is updated
4. Create a pull request with a clear title and description
5. Link any relevant issues in the pull request description
6. Wait for review and address any feedback

## Review Process

1. At least one maintainer must review and approve the pull request
2. Automated tests must pass
3. Code must meet the coding standards
4. Documentation must be updated
5. Changes must not degrade performance (unless justified)

Thank you for contributing to RVV-SIMD!
