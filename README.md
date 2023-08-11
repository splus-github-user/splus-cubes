# splus-cubes
Repository for the stamps and cubes from S-PLUS images

# Requirements

- Python 3.7+
- SExtractor installed either locally or system-wide
- `sewpy` needs to be installed separately following the [authors instruction](https://sewpy.readthedocs.io/en/latest/installation.html)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/splus-collab/splus-cubes.git

```

2. Navigate to the project directory:

```bash
cd splus-cubes
```

- Set up the Python environment:
It is recommended to use a virtual environment to manage project dependencies.
If you don't have virtualenv installed, you can install it by running:


```bash
pip install virtualenv
```

- Create a new virtual environment:
```bash
virtualenv venv
```

- Activate the virtual environment:
```bash
source venv/bin/activate 
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```
This command will read the requirements.txt file and install all the necessary packages for the project.

# Usage

- Print help

```bash
python3 make_scubes_v02.py --help
```

- To create a data cube for a specific galaxy using a S-PLUS field:

```bash
python3 make_scubes_v02.py -v -t 'SPLUS-s28s33' -g 'NGC1365' -l 2400 -i '03:33:36.458 -36:08:26.37' -a 480 -z 0.005476 -p 0.4
```

  - This command will create a cube for the galaxy NGC1365 from the images of the field SPLUS-s28s33. The cube will have 2400x2400 pixels and will be centred  RA = 03:33:36.458 Dec = -36:08:26.37.
    The the major circle to mask the galaxy will start with circle of diameter 480 arcsec and the CLASS_STAR cut is 0.4
