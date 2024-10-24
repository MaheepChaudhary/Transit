# Pythia Suite Deep Analysis.


- There might be activation flux in the models, as a i read in the paper [[sunMassiveActivationsLarge2024]]. Therefore, it would be good for analysis to remove those attention and then see the pattern of activation heads. The lightning of these tokens cannot be said as the token importance is that much. It is to be analysed that activation flux is of first token? and it can be analysed using the self attention pattern difference in the first token compared to others. Also it would be good if we can find out which layer does this behaviour usually occur. 
- Furthermore, the real contribution of token can be find out using the softmax inverse inside the model. In this way we can even track the training procedure of these models. 
- One of the things that could be done is to do intervention using causal analysis, by taking the mean of the embeddings and then sending them. 

## 🐳🐙 TO-DO:
- [ ] Self-attention pattern analysis using Q, K, and V to find the attention in the models. 
- [ ] Read attribution patching, and if not satisfied then find the each token importance in predicting everything, using softmax inverse.
- [ ] Find the activation analysis using the counterfactual analysis to find the impact of the sentence - Find the intervention vector.
- [ ] Find the correlation using covariance matrix between different activations. 
- [ ] Take overall gradient and find the difference between the generated graphs and the graph generated for overall gradient. 


## Analysis of Alpaca Dataset for gradients and activation. 
We generally see a similar pattern emerging in the models and becomes mature overtime. These are the the following patterns that we observe with different scale of models of Pythia ranging from 14m to 1.4B. Below is the summarisation of the observation for different modules of the Pythia models:
### Attention Layer Activations:
The attention layer has most of the activation in the final layer with some noise in other graphs where some activation are high in the end of beginner token. Emergence of two strips in the 1.4B models is one of the things that is strange. Feels like the information extraction has shifted in the last layers form just layer as the number of layers have increased. 
### MLP Layer Activations:
Interestingly, the activations for this layer has an interesting relationship with the residual layer where the activation also activates when the residual layer starts activating but swtich offs quickly, whereas resid activation keep on activating. Similarly, when the resid goes to switch off, the activation of mlp starts activating, creating a **U-shaped pattern**. 
*Can we say that activation's relationship with mlp and resid indicates the resid as memory where the mlp writes in it and retrieves it?*
### Residual Layer Activation:
The residual layer is the sum of the contribution of the mlp and the activation layer, then how come do we see a pattern dominant for the mlp?
The activation of the residual layer follow a pattern of making a block for layers in the middle layers, creating a **D-shaped pattern**.
### MLP Gradient:
It has some obsession with the 3/4th layer, the same layer in which activation explodes in the resid and mlp layer. 
### Attention Gradient:
This indicates typical behaviour where the beginner layer is most activated and the ending layer is activated less, creating a **U-shaped pattern**. 

But, how can these gradients can be interpreted-which index neuron is responsible for generating the output. However, it should be evaluated after taking the overall gradient and finding the analysis of the graph. 

## ⚽️🏈 Counterfactual Activation Analysis:

Here we would need all the activations mean representations, stored in a pickle file. 
These mean activations of all the datasets would be then subtracted from the each layer to get the interevened representations 
which are truly the influence of the dataset. 
This would help in suppressing the the model's common behaviour and focus on the dataset's influence.
Specifically, this would supress the model's attention sink behaviour and abnormal high activation behaviour observed in models. 

We would be using Dynamic Alignment to make the length of activation for all the dataset equal. 
In this format we would be having 3 different lenght means which would be used to subtract the actvation from each dataset activation. 