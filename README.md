# Biped Firmware

## Table of Contents

1. [Project Prerequisites](#project-prerequisites)
2. [Setting Up the Project](#setting-up-the-project)
3. [Building the Project](#building-the-project)
4. [Labs](#labs)

## Project Prerequisites

This project depends on the installation of the following system packages:

Ubuntu:

```bash
apt install curl tar unzip cmake make doxygen python3-serial
```

macOS:
```bash
brew install curl tar unzip cmake make doxygen python3-serial
```

Click [here](https://brew.sh/) on how to install Homebrew on macOS.

Note: CS 431 students are required to develop this project on the lab workstations. In the case where all lab workstations are fully occupied outside the normal lab sections, students are allowed to temporarily set up and develop this project on their personal computers. However, this project is not fully designed nor guaranteed to work on environments other than the lab workstations. Support from the TA for personal computer setups will be limited.

## Setting Up the Project

On the lab workstation, create a project directory under your home document directory and navigate to the created project directory as follows:
```bash
mkdir -pv ~/Documents/Projects
cd ~/Documents/Projects
```

Under the created project directory above, clone this project using `git clone <project-url> biped-firmware`.

After cloning the project, navigate to the project root directory as follows:
```bash
cd ~/Documents/Projects/biped-firmware
```

To set up the project, run the setup script as follows:
```bash
./scripts/biped-firmware/setup.bash
```

During the setup process, pay close attention to the output of the setup script and check for any errors.

Doxygen will be your best friend for this project. To generate the Doxygen documentation for the project, run the Doxygen script as follows:
```bash
./scripts/biped-firmware/doxygen.bash
```

The Doxygen script will attempt to open the generated documentation in the default web browser. To access the generated documentation manually, navigate to `docs/doxygen/html` and open the `index.html` file using a web browser. Please remember, whenever in doubt, always search in Doxygen first. Refer to the [Doxygen Manual](https://www.doxygen.nl/manual/index.html) for more details regarding Doxygen.

## Building the Project

First, navigate to the project build directory as follows:
```bash
cd ~/Documents/Projects/biped-firmware/build/biped-firmware
```

To build the project, perform the following:
```bash
make
```

To facilitate the building process, enable parallel make jobs when using `make` as follows:
```bash
make -j$(getconf _NPROCESSORS_ONLN)
```

Note: the command `getconf _NPROCESSORS_ONLN` above obtains the number of logical CPU cores in the system.

To flash or upload the built project to the Biped, connect the USB-C cable, and then flash the firmware as follows:
```bash
make -j1 upload SERIAL_PORT=/dev/ttyUSB0
```

It is not recommended to use the upload command above to build the project. Instead, combine the build and upload commands together as follows:
```bash
make -j$(getconf _NPROCESSORS_ONLN) && make -j1 upload SERIAL_PORT=/dev/ttyUSB0
```

The serial port might differ depending on the environment and the operating system.

## Labs

All lab instructions and objectives are provided in the form of comment blocks in the project header and source files. Please RTDC (Read the Docs Carefully), as the comment blocks contain crucial information for students to understand and complete the labs.

Check the course Canvas for the demo and codebase submission due dates and details. Certain labs may involve prolonged testing, experiments, and debugging. Therefore, please always start early!

The storage of the lab workstations is entirely local, i.e., everything stays within each of the lab workstations and is not backed up or synced elsewhere. Therefore, the lab groups should stick to one lab workstation as much as possible to avoid repeated setups. Additionally, it is the student's sole responsibility to back up their codebases and employ proper version controls. Any loss of data due to malfunction of lab workstations does not constitute any extension or excuse for any labs.

The labs are newly designed and implemented. Therefore, there are likely rough spots scattered throughout the project. Please help us improve by reporting any bugs, typos, or any other issues or mistakes to the TA.

Refer to the following lab guides:
- [Biped](docs/biped-firmware/general/biped.md)
- [Eclipse](docs/biped-firmware/general/eclipse.md)

Refer to the following lab instructions:
- [Lab 1: Setup and I/O](docs/biped-firmware/labs/lab_1.md)
- [Lab 2: Hardware Timer](docs/biped-firmware/labs/lab_2.md)
- [Lab 3: FreeRTOS](docs/biped-firmware/labs/lab_3.md)
- [Lab 4: I/O Expander](docs/biped-firmware/labs/lab_4.md)
- [Lab 5: Biped Ground Station](docs/biped-firmware/labs/lab_5.md)
- [Lab 6: Sensing and Actuation](docs/biped-firmware/labs/lab_6.md)
- [Lab 7: Control](docs/biped-firmware/labs/lab_7.md)
- [Lab 8: Planning](docs/biped-firmware/labs/lab_8.md)
- [Lab 9: Application](docs/biped-firmware/labs/lab_9.md)

Good luck, and have fun!

Written by Simon Yu.
