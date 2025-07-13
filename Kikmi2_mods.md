MuonClip Optimizer
Without rigor, given an approximately finite pretraining dataset and a fixed model configuration, a more token-efficient optimizer generates more intelligence. Our previous work Moonlight has demonstrated that the Muon optimizer substantially outperforms the widely-used AdamW optimizer for LLM training.
Kimi K2 was designed to further scale up Moonlight, which employs an architecture similar to DeepSeek-V3. Based on scaling-law analysis, we reduce the number of heads for long-context efficiency, and increase MoE sparsity for greater token efficiency. While scaling up, we encountered a persistent challenge: training instability caused by exploding attention logits, an issue that occurs more frequently with Muon but less with AdamW in our experiments. Existing solutions such as logit soft-capping and query-key normalization were found inadequate.
To address this, we introduce the MuonClip optimizer that improves Muon with our proposed qk-clip technique. Specifically, qk-clip stabilizes training by directly rescaling the weight matrices of the query and key projections after Muon updates, thus controlling the scale of attention logits at the source. Concretely, the query and key projections are scaled as follows:
q 
i
​
 =η 
α
 W 
q
​
 x 
i
​
 
k 
i
​
 =η 
1−α
 W 
k
​
 x 
i
​
 
where α is a balancing hyperparameter, so the attention logit becomes:
(η 
α
 q 
i
​
 ) 
⊤
 (η 
1−α
 k 
j
​
 )=ηq 
i
⊤
​
 k 
j
​
 
The adaptive factor η (with threshold t) is set after every step based on the max attention logit in this step:
η=min( 
i,j
max
​
 (q 
i
⊤
​
 k 
j
​
 )
t
​
 ,1)
where t is a pre-set threshold. This is a general technique that can be possibly applied to other stabilization use cases.
Our experiments show that MuonClip effectively prevents logit explosions while maintaining downstream task performance. In practice, Kimi K2 was pre-trained on 15.5T tokens using MuonClip with zero training spike, demonstrating MuonClip as a robust solution for stable, large-scale LLM training.
