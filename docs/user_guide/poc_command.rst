.. _poc_command:

*****************************************
Commmand for Proof Of Concept (POC) Mode
*****************************************

Introduction to the POC Command
===============================

The POC command allows users to try out the features of NVFlare in a proof of concept deployment on a single machine.

Syntax and Usage
=================

.. code-block::

  nvflare poc -h
  
  usage: nvflare poc [-h] [-n [NUMBER_OF_CLIENTS]] [-c [CLIENTS ...]] [-p [SERVICE]] [-e [EXAMPLES]]
                     [-ex [EXCLUDE]] [-gpu [GPU ...]] [-he] [-i [PROJECT_INPUT]] [-d [DOCKER_IMAGE]]
                     [--prepare] [--prepare-examples] [--start] [--stop] [--clean]

  optional arguments:
    -h, --help            show this help message and exit
    -n [NUMBER_OF_CLIENTS], --number_of_clients [NUMBER_OF_CLIENTS]
                          number of sites or clients, default to 2
    -c [CLIENTS ...], --clients [CLIENTS ...]
                          Space separated client names. If specified, number_of_clients argument will be ignored.
    -p [SERVICE], --service [SERVICE]
                          participant, Default to all participants, only used for start/stop poc commands when specified
    -e [EXAMPLES], --examples [EXAMPLES]
                          examples directory, only used in '--prepare' or '--prepare-example' command
    -ex [EXCLUDE], --exclude [EXCLUDE]
                          exclude service directory during --start or --stop, default to , i.e. nothing to exclude
    -gpu [GPU [GPU ...]], --gpu [GPU [GPU ...]]
                          gpu device ids will be used as CUDA_VISIBLE_DEVICES. used for poc start command
    -he, --he             enable homomorphic encryption. Use with '--prepare' command
    -i [PROJECT_INPUT], --project_input [PROJECT_INPUT]
                          project.yaml file path, it should be used with '--prepare' command. If specified, 'number_of_clients','clients' and 'docker' specific options will be ignored.
    -d [DOCKER_IMAGE], --docker_image [DOCKER_IMAGE]
                          generate docker.sh based on the docker_image, used in '--prepare' command. and generate docker.sh '--start/stop' commands will start with docker.sh
    --prepare             prepare poc workspace and provision
    --prepare-examples    create an symbolic link to the examples directory, requires nvflare_example directory with '-e'
    --start               start local
    --stop                stop local
    --clean               clean up poc workspace

.. note::

    After you clone the NVFlare GitHub repository, you may optionally define an ``NVFLARE_HOME`` environment variable to point to the local NVFlare directory so a symbolic link is created to point the transfer directory to the examples in the code base. For example, if the the NVFlare GitHub repository is cloned under ~/projects, then you should set ``NVFLARE_HOME=~/projects/NVFlare``. If the NVFLARE_HOME environment variable is not set, you will need to manually copy the examples to the transfer directory.


Set Up POC Workspace
====================

.. code-block::

  nvflare poc --prepare
  prepare poc at /tmp/nvflare/poc for 2 clients
  This will delete poc folder in current directory and create a new one. Is it OK to proceed? (y/N) y
  provision at /tmp/nvflare/poc for 2 clients with /tmp/nvflare/poc/project.yml
  Generated results can be found under /tmp/nvflare/poc/example_project/prod_00.

This is similar to the old command:

.. code-block::

  poc -n 2

except that the workspace is defaulted to "/tmp/nvflare/poc", and the default of 2 clients does not need to be specified.

.. note::

    Examples were automatically linked from /path_to_NVFlare/NVFlare/examples to /tmp/nvflare/poc/admin/transfer in the output above.
    The transfer directory was set up as a symlink to examples using the NVFLARE_HOME environment variable.


Replace the Default POC Workspace
---------------------------------
You can change the default POC workspace to any location. All you need to do is to set environment variable NVFLARE_POC_WORKSPACE::

    NVFLARE_POC_WORKSPACE="/tmp/nvflare/poc2"

In this example, the default workspace is set to the location "/tmp/nvfalre/poc2".

Now running the following command:

.. code-block::

    nvflare poc --prepare

will generate the POC startup startup kits in the workspace "/tmp/nvfalre/poc2".


Start Package(s)
================
Once the startup kits are generated with the prepare command, they are ready to be started. If you prepared the POC startup kits using default workspace,
then you need to start with the same default workspace, otherwise, you need to specify the workspace.

Start ALL Packages
------------------
Running the following command:

.. code-block::

  nvflare poc --start

will start ALL clients (site-1, site-2) and server as well as FLARE Console (aka Admin Client) located in the default workspace="/tmp/nvflare/poc".

.. raw:: html

   <details>
   <summary><a>Example Output</a></summary>

.. code-block:: none

    start_poc at /tmp/nvflare/poc, gpu_ids=[], excluded = [], services_list=[]
    WORKSPACE set to /tmp/nvflare/poc/example_project/prod_00/site-2/startup/..
    WORKSPACE set to /tmp/nvflare/poc/example_project/prod_00/server/startup/..
    WORKSPACE set to /tmp/nvflare/poc/example_project/prod_00/site-1/startup/..
    PYTHONPATH is /local/custom:
    PYTHONPATH is /local/custom:
    start fl because of no pid.fl
    start fl because of no pid.fl
    start fl because of no pid.fl
    new pid 24462
    new pid 24463
    new pid 24461
    Waiting for SP....
    Waiting for SP....
    2023-07-20 16:29:32,709 - Cell - INFO - server: creating listener on grpc://0:8002
    2023-07-20 16:29:32,718 - Cell - INFO - site-1: created backbone external connector to grpc://localhost:8002
    2023-07-20 16:29:32,718 - Cell - INFO - site-2: created backbone external connector to grpc://localhost:8002
    2023-07-20 16:29:32,719 - ConnectorManager - INFO - 24462: Try start_listener Listener resources: {'secure': False, 'host': 'localhost'}
    2023-07-20 16:29:32,719 - ConnectorManager - INFO - 24463: Try start_listener Listener resources: {'secure': False, 'host': 'localhost'}
    2023-07-20 16:29:32,719 - Cell - INFO - server: created backbone external listener for grpc://0:8002
    2023-07-20 16:29:32,719 - ConnectorManager - INFO - 24461: Try start_listener Listener resources: {'secure': False, 'host': 'localhost'}
    2023-07-20 16:29:32,719 - nvflare.fuel.f3.sfm.conn_manager - INFO - Connector [CH00002 PASSIVE tcp://0:31953] is starting
    2023-07-20 16:29:32,719 - nvflare.fuel.f3.sfm.conn_manager - INFO - Connector [CH00002 PASSIVE tcp://0:22614] is starting
    2023-07-20 16:29:32,720 - nvflare.fuel.f3.sfm.conn_manager - INFO - Connector [CH00002 PASSIVE tcp://0:41710] is starting
    Trying to obtain server address
    Obtained server address: localhost:8003
    Trying to login, please wait ...
    2023-07-20 16:29:33,220 - Cell - INFO - site-1: created backbone internal listener for tcp://localhost:31953
    2023-07-20 16:29:33,220 - nvflare.fuel.f3.sfm.conn_manager - INFO - Connector [CH00001 ACTIVE grpc://localhost:8002] is starting
    2023-07-20 16:29:33,220 - Cell - INFO - site-2: created backbone internal listener for tcp://localhost:22614
    2023-07-20 16:29:33,220 - Cell - INFO - server: created backbone internal listener for tcp://localhost:41710
    2023-07-20 16:29:33,220 - nvflare.fuel.f3.sfm.conn_manager - INFO - Connector [CH00001 PASSIVE grpc://0:8002] is starting
    2023-07-20 16:29:33,220 - nvflare.fuel.f3.sfm.conn_manager - INFO - Connector [CH00001 ACTIVE grpc://localhost:8002] is starting
    2023-07-20 16:29:33,221 - FederatedClient - INFO - Wait for engine to be created.
    2023-07-20 16:29:33,221 - FederatedClient - INFO - Wait for engine to be created.
    2023-07-20 16:29:33,222 - ServerState - INFO - Got the primary sp: localhost fl_port: 8002 SSID: ebc6125d-0a56-4688-9b08-355fe9e4d61a. Turning to hot.
    deployed FL server trainer.
    2023-07-20 16:29:33,229 - nvflare.fuel.hci.server.hci - INFO - Starting Admin Server localhost on Port 8003
    2023-07-20 16:29:33,229 - root - INFO - Server started
    2023-07-20 16:29:33,710 - ClientManager - INFO - Client: New client site-2@192.168.86.53 joined. Sent token: cbb4983f-c895-4364-8508-f58cca53dc31.  Total clients: 1
    2023-07-20 16:29:33,711 - ClientManager - INFO - Client: New client site-1@192.168.86.53 joined. Sent token: e70a1568-2025-4d47-8e64-e3d1a3667a22.  Total clients: 2
    2023-07-20 16:29:33,711 - FederatedClient - INFO - Successfully registered client:site-2 for project example_project. Token:cbb4983f-c895-4364-8508-f58cca53dc31 SSID:ebc6125d-0a56-4688-9b08-355fe9e4d61a
    2023-07-20 16:29:33,712 - FederatedClient - INFO - Successfully registered client:site-1 for project example_project. Token:e70a1568-2025-4d47-8e64-e3d1a3667a22 SSID:ebc6125d-0a56-4688-9b08-355fe9e4d61a
    2023-07-20 16:29:33,712 - FederatedClient - INFO - Got engine after 0.49114251136779785 seconds
    2023-07-20 16:29:33,713 - FederatedClient - INFO - Got the new primary SP: grpc://localhost:8002
    2023-07-20 16:29:33,714 - FederatedClient - INFO - Got engine after 0.49308180809020996 seconds
    2023-07-20 16:29:33,714 - FederatedClient - INFO - Got the new primary SP: grpc://localhost:8002
    Trying to login, please wait ...
    Logged into server at localhost:8003 with SSID: ebc6125d-0a56-4688-9b08-355fe9e4d61a
    Type ? to list commands; type "? cmdName" to show usage of a command.
    > 

.. raw:: html

   </details>
   <br />

.. note::

    If you run ``nvflare poc --start`` before prepare, you will get the following error:

        .. code-block:: shell

           /tmp/nvflare/poc/project.yml is missing, make sure you have first run 'nvflare poc --prepare'

.. note::

    If you run ``nvflare poc --start`` after having already started the server or any of the clients, you will get errors like:

        .. code-block::

            There seems to be one instance, pid=12458, running.
            If you are sure it's not the case, please kill process 12458 and then remove daemon_pid.fl in /tmp/nvflare/poc/server/startup/..

        .. code-block::

            There seems to be one instance, pid=12468, running.
            If you are sure it's not the case, please kill process 12468.

.. note::

    If you prefer to have the FLARE Console on a different terminal, you can start everything else with: ``nvflare poc --start -ex admin``.

Start the server only
----------------------

.. code-block::

    nvflare poc --start -p server

An example of successful output for starting a server:

.. code-block:: none

    WORKSPACE set to /tmp/nvflare/poc/example_project/prod_00/server/startup/..
    start fl because of no pid.fl
    new pid 26314
    2023-07-20 16:35:49,591 - Cell - INFO - server: creating listener on grpc://0:8002
    2023-07-20 16:35:49,596 - Cell - INFO - server: created backbone external listener for grpc://0:8002
    2023-07-20 16:35:49,597 - ConnectorManager - INFO - 26314: Try start_listener Listener resources: {'secure': False, 'host': 'localhost'}
    2023-07-20 16:35:49,597 - nvflare.fuel.f3.sfm.conn_manager - INFO - Connector [CH00002 PASSIVE tcp://0:36446] is starting
    2023-07-20 16:35:50,098 - Cell - INFO - server: created backbone internal listener for tcp://localhost:36446
    2023-07-20 16:35:50,098 - nvflare.fuel.f3.sfm.conn_manager - INFO - Connector [CH00001 PASSIVE grpc://0:8002] is starting
    2023-07-20 16:35:50,100 - ServerState - INFO - Got the primary sp: localhost fl_port: 8002 SSID: ebc6125d-0a56-4688-9b08-355fe9e4d61a. Turning to hot.
    deployed FL server trainer.
    2023-07-20 16:35:50,107 - nvflare.fuel.hci.server.hci - INFO - Starting Admin Server localhost on Port 8003
    2023-07-20 16:35:50,107 - root - INFO - Server started

Start the FLARE Console (previously called the Admin Client)
-------------------------------------------------------------

.. code-block::

    nvflare poc --start -p admin@nvidia.com

Start Clients with GPU Assignment
----------------------------------

The user can provide the GPU device IDs in a certain order, for example:

.. code-block::

    nvflare poc -gpu 1 0 0 2 --start

The system will try to match the clients with the given GPU devices in order. In this example, the matches will be site-1 with GPU_id = 1,
site-2 with GPU_id = 0, site-3 with GPU_id = 0 and site-4 with GPU_id = 2.

If the GPU ID does not exist on the client machine, you will get an error like:

.. code-block:: shell

    gpu_id provided is not available in the host machine, available GPUs are [0]

If no GPU id is specified, the host GPU ID will be used if available.

If there is no GPU, then there will be no assignments. If there are GPUs, they will be assigned to clients automatically.

.. tip::

    You can check the GPUs available with the following command (assuming you have NVIDIA GPUs with drivers installed):

        .. code-block:: shell

           nvidia-smi --list-gpus

Stop Package(s)
===============

To stop packages, issue the command:

.. code-block::

    nvflare poc --stop

Similarly, you can stop a specific package, for example:

.. code-block::

    nvflare poc --stop -p server

Note that you may need to exit the FLARE Console yourself.

Clean Up
========

There is a command to clean up the POC workspace added in version 2.2 that will delete the POC workspaces:

.. code-block::

    nvflare poc --clean
