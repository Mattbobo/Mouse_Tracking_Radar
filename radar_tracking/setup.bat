::setup the environment

::install the python packages "KKT_UI" in dev mode (editable mode) for submodule "UI"
python -m pip install -e ./UI

::install the python packages "KKT_Module" in dev mode (editable mode) for submodule "KKT_Module"
python -m pip install -e ./KKT_Module

::install the python packages "Library" in dev mode (editable mode) for submodule "Library"
python -m pip install -e ./Library
