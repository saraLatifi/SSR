<h1>Streaming session-based Recommendation (VSKNN vs. GAG)</h1>

This is the code and the optimal hyper-parameters used to conduct the experiments in the paper
titled <i>"Streaming Session-Based Recommendation: When Graph Neural Networks meet the Neighborhood"</i>.
It implements streaming session-based VSKNN (VSKNN+), streaming session-based SR (SR+), and their hybrids. Moreover, it
includes the code for the GAG model that is developed and shared by:
<br>
Qiu, Ruihong, Hongzhi Yin, Zi Huang, and Tong Chen,
"GAG: Global attributed graph neural network for streaming session-based recommendation", SIGIR 2020.
(<a href="https://github.com/RuihongQiu/GAG">Original Code</a>)

<b>Note:</b> This is the blinded material for the review process. After the publication, we will share them publicly on GitHub.
<b>Download:</b> You may download all files in one step using the zip file below.

<h2>Datasets</h2>
<div>
We used the pre-processed versions of the <i>Gowalla</i> and <i>Lastfm</i> datasets as shared by the authors of GAG with us. 
    <br>The pre-processed datasets can be downloaded as a zipped file from: 
    <a href="https://drive.google.com/file/d/1dBRsh-isQHKPn8GV5jzQnjEevwxmxx8p/view?usp=sharing">pre-processed datasets</a>
    <br> Then, unzip the file, and copy its content into the folder <i>datasets</i>.
</div>

<h2>Requirements</h2>

To run the VSKNN+ and SR+ models and their hybrids, the following libraries are required:

<ul>
    <li>Anaconda 4.X (Python 3.7)</li>
    <li>NumPy</li>
    <li>Pandas</li>
    <li>Six</li>
    <li>Pympler</li>
    <li>Python-dateutil</li>
    <li>Pytz</li>
    <li>SciPy</li>
    <li>Pyyaml</li>
    <li>Python-telegram-bot</li>
</ul>

Requirements for running the GAG model:

<ul>
    <li>Python 3.8</li>
    <li>Torch</li>
    <li>CUDA 9</li>
</ul>

<h2>Installation (Using Linux)</h2>
<ol>
<li> Download and install Docker (https://www.docker.com/) </li>
<li> Run the following commands: 
<ol>
    <li>For non-neural models (VSKNN+, SR+, and Hybrid):<br>
    <code>docker pull recommendersystems/streaming_session_based_cpu:0.1</code></li>
    <li>For GAG:<br>
    <code>docker pull recommendersystems/streaming_session_based_gpu:0.1</code></li>
</ol></li>
<li> Run the pulled docker image as a container, and map the code directory to the docker container:
<br> 
<ol>
    <li>Using CPU:<br>
    <code>docker run -t -d --name {CONTAINER_NAME} -v {CODE_DIRECTORY}:/project -w /project  {IMAGE_NAME}</code></li>
    <li>Using GPU:<br>
    <code>docker run -t -d --name {CONTAINER_NAME} --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 -v {CODE_DIRECTORY}:/project -w /project  {IMAGE_NAME}</code></li>
</ol> 
</li>
<li> Execute to the running container:<br>
<code>docker exec -it {CONTAINER_NAME} /bin/bash</code></li>
</ol>

<h2>How to Run It</h2>
<ol>
    <h4>Run optimizations or experiments for non-neural models (VSKNN+, SR+, and Hybrid)</h4>
    <ol>
        <li>
            Take the related configuration file <b>*.yml</b> from folder <i>src/baselines/conf/save</i>, and
            put it into the folder named <i>src/baselines/conf/in</i>. It is possible to put multiple configure files
            in this folder. When a configuration file in <i>src/baselines/conf/in</i> has been executed,
            it will be moved to the folder <i>src/baselines/conf/out</i>.
        </li>
        <li>
            Run the following command from the folder <i>src/baselines</i>: 
            <br>
            <code>
                python run_config.py conf/in conf/out
            </code>
        </li>    
        <li>
            Results will be displayed and saved to the results folder (that is determined in the configuration file).
        </li>
    </ol>
    </li>
    <li><h4>Run experiments for GAG</h4>
    Run the <i>main.py</i> file under the folder <i>src/GAG</i> using desired parameters. For example:
        <br>
        <code>
            python -u main.py --dataset='gowalla' --lr=0.001
        </code>
    </li>
</ol>


<h2>Essential Options for a Configuration File</h2>
The following table explains the essential options for a configuration file for optimizing hyper-parameters of a model or running an experiment.
The configuration files used for tuning hyper-parameters and running the experiments of the paper can be found under the folders <i>src/baselines/conf/save/opt</i> and <i>src/baselines/conf/save/exp</i>, respectively.
<br><br>
<div>
<div>
    <table class="table table-hover table-bordered">
        <tr>
            <th width="12%" scope="col"> Entry</th>
            <th width="16%" class="conf" scope="col">Example</th>
            <th width="72%" class="conf" scope="col">Description</th>
        </tr>
        <tr>
            <td>type</td>
            <td>stream</td>
            <td>Values: <b>stream</b> (experiment), <b>stream_opt</b> (hyper-parameters optimization).
            </td>
        </tr>
        <tr>
            <td>evaluation</td>
            <td>evaluation_stream</td>
            <td>Evaluation in terms of the next item of the a session in the current data stream.
            </td>
        </tr>
        <tr>
            <td scope="row">candidate</td>
            <td>5</td>
            <td>Number of test blocks in the candidate set.
            </td>
        </tr>
        <tr>
            <td scope="row">metrics</td>
            <td>-class: accuracy.HitRate<br>
                length: [20]
            </td>
            <td>List of accuracy measures (HitRate and MRR) with the defined cut-off threshold.
            </td>
        </tr>
        <tr>
            <td scope="row">optimize</td>
            <td> class: accuracy.MRR <br>
                length: [20]<br>
                iterations: 100 #optional
            </td>
            <td>Optimization target.
            </td>
        </tr>
        <tr>
            <td scope="row">algorithms</td>
            <td>-</td>
            <td>See the configuration files in the conf folder for a list of the
                algorithms and their parameters.<br>
            </td>
        </tr>
    </table>
</div>
</div>

