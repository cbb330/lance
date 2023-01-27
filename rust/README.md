# Rust Binding for Lance Data Format

> :warning: **Under heavy development**

## Introduction

<div align="center">
<p align="center">

<img width="257" alt="Lance Logo" src="https://user-images.githubusercontent.com/917119/199353423-d3e202f7-0269-411d-8ff2-e747e419e492.png">

**Blazing fast exploration and analysis of ML data using python and SQL, backed by an Apache-Arrow compatible data format**
</p></div>

## Support Matrix in Rust Bindings

|                   | Read | Write | Null |
|-------------------|------|-------|------|
| Plain             | Yes  | Yes   | No   |
| Var-length Binary | Yes  | Yes   | Yes  |
| Dictionary        | Yes  | Yes   | No   |
| RLE               | No   | No    | No   |


## Python integration

Done via pyo3 under `pylance` directory (still `import lance` module name though)