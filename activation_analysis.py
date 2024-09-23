from imports import *

class act_analysis_graph:
    
    def __init__(self):
        self.model_names = ["Pythia14m", "Pythia70m", "Pythia160m", "Pythia410m", "Pythia1b", "Pythia1.4b"]
    
    def forward(self):
        
        for model_name in self.model_names:
            
            with open(f"data/tinystories/{model_name}/activation_attn.pkl", "rb") as f:
                data = pickle.load(f)
                data = np.linalg.norm(data, axis = 1)
                self.plot(data, model_name, "tinystories", element_name="attn")
            
            with open(f"data/tinystories/{model_name}/activation_mlp.pkl", "rb") as f:
                data = pickle.load(f)
                data = np.linalg.norm(data, axis = 1)
                self.plot(data, model_name, "tinystories", element_name="mlp")
            
            with open(f"data/tinystories/{model_name}/activation_resid.pkl", "rb") as f:
                data = pickle.load(f)
                data = np.linalg.norm(data, axis = 1)
                self.plot(data, model_name, "tinystories", element_name="resid")
            
        for model_name in self.model_names:
            
            with open(f"data/alpaca/{model_name}/activation_attn.pkl", "rb") as f:
                data = pickle.load(f)
                data = np.linalg.norm(data, axis = 1)
                self.plot(data, model_name, "alpaca", element_name="attn")
            
            with open(f"data/alpaca/{model_name}/activation_mlp.pkl", "rb") as f:
                data = pickle.load(f)
                data = np.linalg.norm(data, axis = 1)
                self.plot(data, model_name, "alpaca", element_name="mlp")

            with open(f"data/alpaca/{model_name}/activation_resid.pkl", "rb") as f:
                data = pickle.load(f)
                data = np.linalg.norm(data, axis = 1)
                self.plot(data, model_name, "alpaca", element_name="resid")
        
        for model_name in self.model_names:
            
            with open(f"data/summarisation/{model_name}/activation_mlp.pkl", "rb") as f:
                data = pickle.load(f)
                data = np.linalg.norm(data, axis = 1)
                self.plot(data, model_name, "summarisation", element_name="mlp")
            
            with open(f"data/summarisation/{model_name}/activation_resid.pkl", "rb") as f:
                data = pickle.load(f)
                data = np.linalg.norm(data, axis = 1)
                self.plot(data, model_name, "summarisation", element_name="resid")
            
            with open(f"data/summarisation/{model_name}/activation_attn.pkl", "rb") as f:
                
                data = pickle.load(f)
                data = np.linalg.norm(data, axis = 1)
                self.plot(data, model_name, "summarisation", element_name="attn")
    
    def plot(self, data, model_name, dataset_name, element_name):
        
        y_values = np.log(data)
        x_values = ["layer " + str(i) for i in range(len(data))]

        # Create scatter plot
        fig = go.Figure(data=go.Scatter(
            x=x_values,
            y=y_values,
            mode='markers',
            marker=dict(
                showscale=False  # Do not show color scale
            )
        ))

        # Customize the layout
        fig.update_layout(
            title=f"[{model_name}-{dataset_name} (7 words/string)] Activation Avg. Norm. vs Layer for {element_name}",
            xaxis_title='Layer Name in Pythia Model',
            yaxis_title='Average Norm of Activation (Log-Scale)',
            showlegend=False
        )

        fig.write_image(f"figures/{dataset_name}/{model_name}/activation_analysis_{element_name}.png")

if __name__ == "__main__":
    
    act_analysis = act_analysis_graph()
    act_analysis.forward()