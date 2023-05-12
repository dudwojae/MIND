# MIND: Masked and Inverse Dynamics modeling for Data-Efficient Deep Reinforcement Learning
This repository provides the code to implement the MIND.

# File Description
    .
    ├── agents
    │   └── mind_agent.py         # The agent script to select actions and optimize polices
    ├── environment                     
    │   ├── env.py                # Atari environment
    ├── networks                     
    │   ├── encoder.py            # Deep neural networks code needed to train MIND
    │   ├── modules.py            # Deep neural network modules code needed to train MIND
    ├── tasks                     
    │   ├── mind.py               # Code to train or test MIND
    │   ├── mind_test.py          # Code used when testing in mind.py
    ├── utils                    
    │   ├── args.py               # Arguments needed to run the code
    │   ├── augmentation.py       # Masking augmentation code
    │   ├── loss.py               # Inverse dynamics loss
    │   ├── memory.py             # Prioritized experience replay
    │   ├── mypath.py             # The path to saver or load the file
    │   ├── weight_init.py        # Deep neural networks initialization
    └── run_mind.py            # The main run code
    
# Installation
The python version we used is 3.6.13.
~~~
pip install -r requirements.txt
~~~

# Train MIND
~~~
python run_mind.py
~~~

