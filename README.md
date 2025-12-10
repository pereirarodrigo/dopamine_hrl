<div align="center">
    <img src="img/logo.png" width="350px" height="150px"></img><br>
</div>

<br>

The **Dopamine-HRL Research System**, or simply **D-HRL**, is a software suite that enables professionals to simulate dopamine-influenced decision-making in risk-based settings. It is built upon a computerised version of the Iowa Gambling Test (IGT), and leverages a hierarchical reinforcement learning (HRL) agent, coupled with a simulated dopaminergic module, to model agent dynamics.

## About

This project is the culmination of my Master of Science (MSc) dissertation, "Simulating Risk-Based Decision-Making and Dopaminergic
Dysfunctions Using Hierarchical Reinforcement Learning", undertaken at the University of Essex Online and supervised by [Professor Diego Navarra](https://www.linkedin.com/in/diego-navarra-800228/). The project aimed to investigate whether a biologically inspired HRL model, itself based on reinforcement learning (RL), would:

* Produce agents that generate responses reasonably similar to that of humans in risk or uncertain-based situations
* Capture the impact of simulated dopamine depletion or overactivity in decision-making performance
* Display better performance than traditional, non-hierarchical RL approaches

It was released to the public as a gesture of support for open-source systems and general open science, and also out of a desire to contribute to advancements in neuroscience, cognitive science, and artificial intelligence (AI). Therefore, it remains fully open to modifications, improvements, extensions, and further research efforts.

## Features

* A novel, hybrid dopamine-HRL agent based on joint neuroscientific-AI findings
* A computerised, modular, and extensible reinforcement learning-compatible IGT environment
* Baseline models for performance comparisons
* Dopaminergic configurations meant to mimic dopamine-depleted and dopamine-overactive conditions
* IGT dataset handling utilities (specific to those from Steingroever et al. (2014))
* An integrated suite that can run simulations, generate a summary, and execute statistical analyses against human-originated IGT data
* Chart generation to enhance statistical interpretability (available for summary *and* agent vs. human analyses)
* Easy access to model hyperparameters and simulation parameters, offering full customisability
* Docker and web interface support

## How to Use

Before doing anything, make sure to download the data from [Steingroever et al. (2014)](https://osf.io/8t7rm/overview), which constitutes the core of the system's evaluation procedures. Once done, extract the contents of the downloaded folder to `dopamine_hrl/datasets/steingroever`.

To run the system itself, you have two available options: Docker, or a local web interface. 

### Docker Path

Docker is the most straightforward path given it builds app images that are self-contained, though it does necessitate that you have Docker installed beforehand. To run the system as a Docker image, run the following command:

```cmd
docker built -t <app_image_name> .
```

where `<app_image_name>` is a name of your choice to the system image. After creating the Docker image, simply run it with the following command:

```cmd
docker run -it -p 8501:8501 <app_image_name>
```

After this, the web interface should be accessible at https://localhost:8501 and the system should be ready for use.

### Local Path

Should you wish, however, to run the system locally, please take note of the following steps:

1. Install the `uv` Python package via the following command:

```cmd
pip install uv
```

2. Navigate to the `dopamine_hrl` folder and activate the system's `uv` environment by typing the following command:

```cmd
source .venv/bin/activate
```

3. Afterwards, run the following command to start the app:

```cmd
uv run streamlit run app.py
```

The system should now be online and accessible at https://localhost:8501.

## Acknowledgements

This project wouldn't have been possible without all the extensive prior work focusing on psychological tests, dopaminergic effects on decision-making, and HRL. Papers such as those from [Sutton et al. (1999)](https://doi.org/10.1016/S0004-3702(99)00052-1), [Mehler-Wex et al. (2006)](https://doi.org/10.1007/BF03033354), [Frank and Claus (2006)](https://doi.org/10.1037/0033-295X.113.2.300), [Dabney et al. (2020)](https://doi.org/10.1038/s41586-019-1924-6), [Steingroever et al. (2014)](https://openpsychologydata.metajnl.com/articles/10.5334/jopd.ak) and many others were critical in shaping the research direction of this project and lending academic weight to the overall structure leveraged here.

Furthermore, and arguably above all academic work leveraged in this research, the tremendous support from Professor Navarra must be acknowledged. His insights, questions, and suggestions proved instrumental in transforming this project from a mess of neuro-focused ideas into a coherent mix of cognitive AI and computational principles.  