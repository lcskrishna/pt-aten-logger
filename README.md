# pt-aten-logger

This tool is used for logging some information out of PyTorch workloads. Can be used for dumping shapes, operator level time.

## Setup

To install and use the tool, use the below:

### Installation steps
```
git clone https://github.com/lcskrishna/pt-aten-logger.git
cd pt-aten-logger
python setup.py install
```

### Usage
Please use the following context manager to dump out the operator level information.

```
import pt_aten_logger
from pt_aten_logger import ATenShapeDtypeDumpInfo 

with ATenShapeDtypeDumpInfo():
   output = model.forward(inputs)
   ...
```

The above dumps out the operator level information in the stdout. Please look through examples/logs folder for sample outputs. 

