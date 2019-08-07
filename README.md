# flowering-data-gen
Generate expression data according to the ODE model in [Jaeger et al. 2013](https://doi.org/10.1105/tpc.113.109355).

## Usage
1. Install packages in requirements.txt
```bash
pip install -r requirements.txt
```
2. Generate network files
```bash
python jaeger.py
```
3. (Optional) Reproduce Figure S2 in Jaeger et al.
```bash
python -c "import jaeger; jaeger.reproduce_figure_s2()"
```
