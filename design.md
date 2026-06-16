\section{Insight}

\begin{figure}[t]
\centerline{\includegraphics[width=\linewidth]{figures/tradeoff.pdf}}
\caption{Memory \& communication relationship model in data parallelism, where $\omega$ represents the communication required for a global \texttt{AllReduce}. ZeRO-2 requires 1 \texttt{AllGather} and 1 \texttt{ReduceScatter} in one training iteration, while ZeRO-3 requires 2 \texttt{AllGather} and 1 \texttt{ReduceScatter}.}
\label{fig: tradeoff}
\end{figure}

\subsection{Parameter Data Reusability} \label{tradeoff}
When using DP and its variants for distributed LLM training, there exists a trade-off between memory utilization and communication cost. Fig.~\ref{tradeoff} illustrates the schematic relationship of trade-offs for various methods in the memory-comm 2D space. ZeRO-2, ZeRO++, MICS, and ZeRO-3 are iterative methods where their different partitioning granularities—or in other words, the redundancy of parameter copies—essentially avoid corresponding \allgather{} communication by storing parameters. Among them, ZeRO-2 and ZeRO-3 can be viewed as the two extremes of the trade-off between caching and communication. ZeRO-2 maintains 100\% parameter redundancy (full copies) across nodes, thereby achieving training with minimal communication cost. ZeRO-3 achieves zero redundancy by scattering the model across the entire cluster, ensuring high memory efficiency. The transition zone between the two arises from the extra \allgather{} operation ZeRO-3 performs during the backward propagation phase to collect parameters, where the parameters collected by \allgather{} are identical to those in the forward phase. To this end, if partial parameters can be cached during the forward propagation and reused in the backward propagation, the intervening \allgather{} communication overhead is saved.


For a parameter cache of size $u$, it can save $u(d-1)$ communication volume in \allgather{}, thereby accelerating the overall training by:
\begin{equation}
T_{\text{acc}} = \frac{(d-1)}{bandwidth} u
\label{eq1}
\end{equation}
This linear relationship implies that such reusability is always feasible, allowing any amount of memory surplus to be converted into training acceleration. It is worth noting that due to the synchronous nature of collective communication, distributed devices must ensure cache consistency to effectively eliminate communication. For example, in Fig.~\ref{zero}, if we wish to cache the parameters of S3, combining the local shards of each GPU, GPU0 must retain shards 8 and 9, while GPU1 retains shards 7 and 9, and GPU2 retains shards 7 and 8.

\subsection{Structural Memory Surplus}
During the model training process, memory primarily stores model states and activation values. Among these, model states (parameters, gradients, and optimizer parameters) can be calculated accurately given known hyperparameters. Activation values refer to the intermediate variables calculated during forward propagation that are needed during backward propagation to compute gradients. When \texttt{checkpointing} is enabled, activation values only include the input of each layer, calculated as:
\begin{equation}
\text{M}_{\text{act\_ckpt}} \approx L \times BS \times S \times H \times P
\label{eq2}
\end{equation}
where notates L(Layers), S(Sequence length), H(Hidden size) and P(Precise bytes).
$\text{M}_{\text{act\_ckpt}}$ is proportional to batch size ($BS$). Therefore, when model states cannot fully occupy memory, fine-tuning batch size is the most feasible and general method, capable of intuitively converting memory resources into throughput. However, this fine-tuning is coarse-grained in LLM training; taking Llama3 70B \cite{dubey2024llama} as an example ($L=80, H=8192, S=4096, BS=1, P=2$), for every increment $BS=BS+1$, memory usage may increase by approximately 5.4GB. Adjustments at this level are very prone to leaving memory fragments ($\text{size} < 5.4\text{GB}$); these fragments are structural issues stemming from the model scale and are difficult to utilize. Additionally, there is an upper limit to improving system throughput by increasing BS; when compute units are saturated, the resulting benefits gradually decline, and an excessively high BS may lead to model overfitting problems.


\subsection{Motivation}
Therefore, this paper urgently seeks to combine the above two insights to design a method that enables GPU memory to be fully and losslessly utilized. Specifically, we hope the resulting method allows the GPU to fully cache model parameters, assisting ZeRO-3 in achieving cost-free performance improvements through the aforementioned memory-communication conversion. 
% Fig.~\ref{fig6} shows diagram illustrating the memory surplus utilization in \sysname{}.
Fig.~\ref{fig1}(a) and (b) demonstrate the performance of our method in critical situations. In scenarios where structural memory surplus exists in ZeRO-3 (as shown in Fig.~\ref{fig: tradeoff}), our method utilizes memory almost completely and provides higher throughput. In scenarios where memory is ample, our method can also adaptively maintain memory utilization at a high water mark without fine-tuning BS.

\section{Design} \label{design}

\begin{figure*}[htbp]
\centerline{\includegraphics[width=\linewidth]{figures/overview.png}}
\caption{Overview of \sysname: An adaptive cache is logically added to ZeRO-3 (FSDP) distributing procedure to achieve data (module or shards) caching between forward and backward progrations. A detection is performed in initial iterations to select system information. By profiling the distributing behaviour, CITs initials shards with local (-1) or collected (0) labels. The scheduler solve the optimization problem according to the gain equation formulated by relative factors, and finally modifies CITs with cache (1) labels. CITs instruct devices to deal with shards in subsequent iterations.}
\label{overview}
\end{figure*}

\subsection{System Overview}
We present \sysname, a communication-efficient distributed training method designed to exploit structural memory surplus in large-scale model training. As illustrated in Fig.~\ref{overview}, the system operates as a transparent optimization tier atop standard Fully Sharded Data Parallelism (FSDP) protocol. To avoid potential underutilization of memory caused by the "fetch-compute-discard" process in the protocol, our system introduces a stateful caching mechanism governed by a Profile-Guided Static Scheduling paradigm. When evaluating the gain of caching a module, we consider the size gain from \S \ref{tradeoff}, as well as two other types of gains: operator dependency and global lifespan (\S \ref{cost model}).

The workflow is orchestrated in three distinct phases:

\textbf{Profiling Phase (\S \ref{memory profiler} and \S \ref{cost model}):} During the initial iterations of training, the system profiles its state without interfering with FSDP operation. The state includes stage distribution, communication latency, computational complexity, and most importantly, GPU memory surplus. This state is used to calculate the caching cost and is fed into the scheduler for evaluation and subsequent cache management.

\textbf{Planning Phase (\S \ref{scheduler}):} A centralized scheduler analyzes the profiling state to construct a deterministic execution plan. The scheduler solves a resource-constrained optimization problem to identify the optimal set of modules to persist in memory. The output is presented as a set of Cache Instruction Tables (CITs), which guide the storage management of submodules during runtime.

\textbf{Execution Phase (\S \ref{runtime engine}):} Distributed workers execute the training loop according to the instruction tables, dynamically managing the lifecycle of computable modules and ZeRO-shards to eliminate redundant communication and satisfy memory surplus. To fully mine available memory, Madeline uses ZeRO-shards to fill fine-grained memory fragments after first caching complete Model Modules at a coarse granularity.

\subsection{Memory Profiler} \label{memory profiler}
To safely utilize surplus memory, we rigorously quantify available resources and the distinct costs associated with caching different modules.
\subsubsection{Memory Characterization} We define the total device memory capacity as $M_{dev}$. During the training process at time step $t$, memory consumption is composed of three parts:
\begin{equation}
    M_{used}(t) = M_{static} + M_{activation}(t) + M_{temp}(t)
\end{equation}

where $M_{static}$ denotes persistent states, $M_{activation}(t)$ represents the cumulative size of activations, and $M_{temp}(t)$ accounts for temporary collective communication buffers (fixed \textit{bucket}). The Structural Surplus Memory available for caching module $u$ is defined as the safe margin between peak memory usage and the device limit over its required lifespan $T_u$:
\begin{equation}
    M_{surplus}^{(u)} = M_{dev} - \max_{t \in T_u} (M_{used}(t)) - \epsilon
\end{equation}

where $T_u$ is the time interval for processing module $u$, and $\epsilon$ is a safety buffer to handle memory fragmentation.

\subsection{Cost Model} \label{cost model}
\subsubsection{Gain Modeling} Madeline prioritizes caching complete \textit{model modules}, as these modules possess computational significance compared to thoroughly partitioned ZeRO-shards. Since current training systems typically use different CUDA streams to control computation and communication respectively, caching computable modules makes sense for computation-communication overlap (e.g. Fig.~\ref{dependency} (1) and (3)); this overlap can further enhance the benefits of caching. However, on top of this additional "overlap gain," the efficiency of caching a module is various. We observe a critical factor influencing the potential gain of caching:

\textbf{Local Dependency Gain (Intra-Stage):} This is similar to a Head-Of-Line blocking problem. Within a single communication \textit{bucket} (Stage), backward computation is sequential ($u_{tail} \to u_{head}$). Although a cached tail module still reduces the amount of data in the AllGather operation, its effective gain is not significant because its computation is blocked by the dependency chain of the computation graph, as shown in Fig.~\ref{dependency} (2); In contrast, a cached head module can help overlap the backward computation and the \allgather{} communication, therefore improve gain additionally, as shown in Fig.~\ref{dependency} (3).

\textbf{Global Lifespan Gain (Inter-Stage):} modules closer to the input layer (Stage 1) must be held in memory for a longer duration (from forward pass to the end of backpropagation). This high "Time-Space Integral" represents a higher global opportunity cost compared to output layers.
Due to the accumulation of activation, the memory usage increases before the loss function and reaches its peak after the first \allgather{} operation in the backpropagation (at which point an  $M_{temp}$ is stored incrementally). Prioritizing the storage of modules at the end of the network can mitigate this peak and stay away from potential OOM error.

We encapsulate these factors into a unified Gain ($G(u)$) formulation. We model the backward pass of a bucket $B_k$ as an ordered sequence $U_k = \{u_{k,n}, \dots, u_{k,1}\}$, where $u_{k,n}$ is the first executed module (tail module). The Gain of caching module $u$ is defined as:
\begin{equation}
    G(u) = \underbrace{S(u)}_{\text{Bandwidth Gain}} + \underbrace{( \frac{k \cdot D}{bs} )^\alpha}_{\text{Lifespan Gain}} + \underbrace{(1-\frac{\text{pos}(u)}{n} )^\beta}_{\text{Latency Gain}}
    \label{eq: gain}
\end{equation} 


Where $S(u)$ is the size of module $u$, representing the deterministic reduction in communication volume. $\text{pos}(u)$ represents the execution index $\in \{1,...n\}$. $bs$ is the \allgather{} bucket size. $\alpha,\beta$ are hyper-parameters balancing bandwidth saving and latency hiding.
This formula mathematically penalizes head modules ($\text{pos} \to 1$). Since head units cannot unlock the pipeline head, their latency gain decays quadratically, reflecting their lower contribution to pipeline parallelism compared to the tail modules ($\text{pos} \to n$).
\begin{figure}[t]
\centerline{\includegraphics[width=\linewidth]{figures/dependency.png}}
\caption{Module-level Backward Scheduling of simplified ZeRO-3 (FSDP) methods. Module a, b and c are all-gathered and then computed, and finally do Reduce-Scatter on gradients. 1) Dependency between communication and computation operations. 2) Although module b is cached in advance, it cannot be executed without module a's output. 3) With module a cached, the computation is not dependent to Allgather, and can be overlapped with it.}
\label{dependency}
\end{figure}

\subsection{The Adaptive Scheduler} \label{scheduler}
The core of our design is the Adaptive Scheduler. Based on the profiled costs and modeled gains, we transform the scheduling challenge into a deterministic optimization problem to maximize system throughput.

\subsubsection{Problem Formulation}
We formulate the selection of cached modules as a 0/1 Knapsack Problem.
Let $\mathcal{U} = \{u_1, u_2, \dots, u_N\}$ be the set of all candidate modules in the model.
Let $x_i \in \{0, 1\}$ be the cache action, where $x_i=1$ denotes caching module $u_i$.
Let $S_i$ be the size (individual capacity cost) of module $u_i$.
Let $W_{avail}$ be the total surplus memory capacity available for caching.
Our objective is to select a subset of modules to maximize the total Gain, subject to the capacity constraint:
\begin{equation}
    \begin{aligned} \text{max} \quad & \sum_{i=1}^{N} G(u_i) \cdot x_i \\ \text{s.t.} \quad & M_{used}(t) + \sum_{u \in \mathbb{U}_t} (S(u_i) \cdot x_i) \le M_{dev}\end{aligned}
\end{equation}
where $\mathbb{U}_t$ includes the accumulated modules at time step $t$.

Based on direct modeling, the problem solved is a combinatorial optimization knapsack problem. However, within a training step, the fluctuation of $M_{peak}^{(t)}$ over time adds time-variance to the constraints, making it an NP-hard problem with high solution complexity. Therefore, simplification of the problem is necessary. By analyzing the mechanism during the training step and the memory monitoring in real-world tests, it is known that the time fluctuation of $M_{peak}^{(t)}$ follows a regular pattern: it rises in a sawtooth manner during forward propagation (accumulation of activation values), reaches a peak after loss function calculation, and then decreases in stages during backward propagation (release of activation values and gradient computation). Therefore, the above constraint can be simplified to:

\begin{equation}
    \sum_{i=1}^{N} S(u_i) \cdot x_i \le M_{surplus}^{(u_N)}
\end{equation}

Thus, any set of cached weights that can be stored in memory at the moment of peak memory pressure ($t_{end}$) can also be successfully stored at any prior moment $t < t_{end}$, because at any earlier moment $t$, the available remaining memory is strictly greater than $M_{surplus}^{(u_N)}$. At this point, the problem is simplified to the \textit{0-1 Knapsack problem}, which is an NP-complete problem.

\subsubsection{Optimization Solver}
Since we have reduced the problem to a classic 0/1 Knapsack problem with discrete weights, we can solve it optimally using \textit{Dynamic Programming} rather than relying on approximate greedy heuristics or proving NP-Hardness. We define a DP state array $dp[m]$, representing the maximum Gain achievable with a total cache occupancy of exactly $m$. The state transition equation is:
\begin{equation}
    dp[m] = \max(dp[m], \ dp[m - S(u_i)] + G(u_i))
\end{equation}

\subsubsection{Algorithm Complexity.} The algorithm iterates through each module $u_i$ and updates the gain for valid capacities $m$ from $M_{surplus}^{(u_N)}$ down to $S(u_i)$.

\textbf{Time Complexity:} $O(N \cdot M)$, where $N$ is the number of modules and $W$ is the discretized capacity of $M_{surplus}^{(u_N)}$.

\textbf{Efficiency:} Given that $N$ is typically in the hundreds (layers) and $M$ can be limited (e.g., 10GB), the solution space is small. This allows the solver to find the global optimum in milliseconds during the Planning Phase, introducing negligible overhead to the training pipeline.

\textbf{Output.} The solver outputs the optimal configuration $\{x_i^*\}$, which is compiled into the Cache Instruction Tables (CITs). These tables are distributed to workers, dictating precisely which modules are retained at the moment of peak pressure; CITs also guarantee both memory safety and maximum communication reduction.

\subsection{Runtime Execution Engine} \label{runtime engine}
The execution engine is responsible for enforcing the CITs generated by the solver. It is implemented as a lightweight interception layer injected into the distributed training runtime.

\textbf{State Management.} The engine maintains a Caching State Machine for each module, which queries the CIT using the current module ID. If the cache action $x_i = 0$: Upon completion of the forward propagation, standard behavior triggers a release mechanism to free the full parameter tensor. If the cache action $x_i = 1$: which means a cache decision, the engine intercepts the release signal and retains the memory handle of the gathered parameters.

\textbf{Lifecycle Extension.} In the backward pass, when the runtime prepares to execute a module, the engine first checks the local cache. If the data is present (Hit), it bypasses the communication backend, serving the parameters directly from memory via zero-copy reference. This allows computation to start immediately, potentially overlapping with the pre-fetching of subsequent stages. Once the backward propagation for that unit is complete, the engine strictly enforces memory release.

\textbf{Gap Filling.} The Madeline scheduler first utilizes memory at a coarse granularity by caching complete modules. After the CIT is confirmed, the knapsack problem solution is likely to leave a small amount of residual memory fragments unfilled; the engine will then use ZeRO shards (which are unstructured tensor data that cannot be used for computation and only save communication volume) to fill other spaces at a fine granularity if needed. These shards are equally applicable to Equation 1 and contribute to communication reduction; which specific shards are selected for caching at this stage is not important.


