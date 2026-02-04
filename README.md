# ThermoOpt

``ThermoOpt`` is a Python package for the modeling and optimization of thermodynamic cycles.

📚 **Documentation**: [https://turbo-sim.github.io/thermopt/](https://turbo-sim.github.io/thermopt/) *(under construction)*  
📦 **PyPI package**: [https://pypi.org/project/thermopt/](https://pypi.org/project/thermopt/)


## 🚀 User installation (via PyPI)

If you just want to use ``ThermoOpt``, the easiest way is to install it from PyPI:

```bash
pip install thermopt
```


You can then verify the installation with:

```bash
python -c "import thermopt; thermopt.print_package_info()"
```


## 🛠️ Developer installation (from source with Poetry)

This guide walks you through installation for development using `Poetry`, which manages both dependencies and virtual environments automatically.


1. **Install Poetry package manager**
   Follow the official guide: [Poetry Installation](https://python-poetry.org/docs/#installation)  
   Then verify the installation:
   ```bash
   poetry --version
   ```

2. **Clone the repository from GitHub**
   ```bash
   git clone https://github.com/turbo-sim/thermopt.git
   ```

3. **Navigate to the project directory**
   
   ```bash
   cd thermopt
   ```

4. **Install the package using Poetry**
   
   ```bash
   poetry install
   ```

5. **Verify the installation**
   
   ```bash
   poetry run python -c "import thermopt; thermopt.print_package_info()"
   ```
   
   If the installation was successful, you should see output similar to:
   
   ```
   --------------------------------------------------------------------------------
         ________                        ____        __
        /_  __/ /_  ___  _________ ___  / __ \____  / /_
         / / / __ \/ _ \/ ___/ __ `__ \/ / / / __ \/ __/
        / / / / / /  __/ /  / / / / / / /_/ / /_/ / /_
       /_/ /_/ /_/\___/_/  /_/ /_/ /_/\____/ .___/\__/
                                          /_/
   --------------------------------------------------------------------------------
   --------------------------------------------------------------------------------
    Version:       0.2.2
    Repository:    https://github.com/turbo-sim/thermopt
    Documentation: https://turbo-sim.github.io/thermopt/
   --------------------------------------------------------------------------------
   ```


## 📂 Examples

The [examples](examples) directory contains a variety of ready-to-run thermodynamic cycle cases, covering different working fluids and applications.

Each example:
- Is defined in a `.yaml` input file
- Is executed via a corresponding `run_optimization.py` script
- Outputs results in a subdirectory called `results/`

To run any example, navigate to the corresponding subfolder and execute the optimization script.