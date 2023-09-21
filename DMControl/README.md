# File Description
    .
    ├── agents
    │   └── mind_dmc_agent.py         # The agent script to select actions and optimize polices
    ├── networks                     
    │   ├── encoder.py                # Deep neural networks code needed to train MIND
    │   ├── modules.py                # Deep neural network modules code needed to train MIND
    ├── tasks                     
    │   ├── mind_dmc.py               # Code to train or test MIND
    │   ├── mind_dmc_test.py          # Code used when testing in mind_dmc.py
    ├── utils                    
    │   ├── args.py                   # Arguments needed to run the code
    │   ├── augmentation.py           # Masking augmentation code
    |   ├── logger.py                 # Various metrics logger
    │   ├── loss.py                   # Inverse dynamics loss
    │   ├── memory.py                 # Experience replay
    │   ├── mypath.py                 # The path to saver or load the file
    │   ├── recorder.py               # Video recorder
    │   ├── scheduler.py              # Learning rate scheduler
    │   ├── weight_init.py            # Deep neural networks initialization
    └── run_mind_dmc.py               # The main run code
    
# Installation
The python version we used is 3.7.11.
~~~
pip install -r requirements.txt
~~~

# Train MIND
~~~
python run_mind_dmc.py
~~~
