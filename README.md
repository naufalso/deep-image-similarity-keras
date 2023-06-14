# Deep Image Similarity with TensorFlow Keras

[WORK IN PROGRESS]

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A TensorFlow Keras implementation of a deep learning mode. for image similarity using a Siamese Network

This project provides a simple and flexible framework for building and training deep learning models to measure the similarity between images.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Training](#training)
- [Inference](#inference)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Requirements

To use this project, you will need to have the following dependencies installed:

- TensorFlow (version 2.11.0)
- TensorFlow AddOns (version 0.20.0)
- TensorFlow Hub (version 0.13.0)
- Other dependencies...

## Usage

To use the deep image similarity model, follow these steps:

1. Clone the repository:
```shell
git clone https://github.com/naufalso/deep-image-similarity-keras
cd deep-image-similarity-keras
```

2. Install the depencencies:
```shell
pip install -r requirements.txt
```

3. Download the dataset (see [Dataset](#dataset) section). Split the dataset into train, validation, and test using the following scripts:
```shell
python3 dataset_splitter.py [/path/to/dataset] [/path/to/output] [split_ratio:60,20,20]
```

4. Train the image similarity model (see [Training](#training) section). Use the following scripts as the minimal sample:
```shell
python3 train.py --model_config [/path/to/model_config] --dataset_path [/path/to/splitted_dataset]
```

5. Inference [TBD]
```shell
[TBD]
```


## Dataset

[TBD]

## Training

[TBD]

## Inference

[TBD]


## Model Architecture

[TBD]

## Results

[TBD]

## Contributing

Contributions are welcome! If you would like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature/bug fix.
3. Make your changes and commit them.
4. Push your changes to your fork.
5. Submit a pull request.

Please ensure that your code adheres to the project's coding conventions and includes appropriate documentation and tests.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.