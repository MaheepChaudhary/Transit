from imports import *

class act_analysis_graph:
    
    def __init__(self):
        self.model_names = ["Pythia14m", "Pythia70m", "Pythia160m", "Pythia410m", "Pythia1b", "Pythia1.4b"]
    
    def forward(self):
        
        for model_name in self.model_names:
            
            with open(f"data/tinystories/{model_name}/activation_attn.pkl", "rb") as f:
                data = pickle.load(f)
                data = np.linalg.norm(data, axis = 1)
                data = data/np.max(data)
                self.plot(data, model_name, "tinystories", element_name="attn")
            
            with open(f"data/tinystories/{model_name}/activation_mlp.pkl", "rb") as f:
                data = pickle.load(f)
                data = np.linalg.norm(data, axis = 1)
                data = data/np.max(data)
                self.plot(data, model_name, "tinystories", element_name="mlp")
            
            with open(f"data/tinystories/{model_name}/activation_resid.pkl", "rb") as f:
                data = pickle.load(f)
                data = np.linalg.norm(data, axis = 1)
                data = data/np.max(data)
                self.plot(data, model_name, "tinystories", element_name="resid")
            
        for model_name in self.model_names:
            
            with open(f"data/alpaca/{model_name}/activation_attn.pkl", "rb") as f:
                data = pickle.load(f)
                data = np.linalg.norm(data, axis = 1)
                data = data/np.max(data)
                self.plot(data, model_name, "alpaca", element_name="attn")
            
            with open(f"data/alpaca/{model_name}/activation_mlp.pkl", "rb") as f:
                data = pickle.load(f)
                data = np.linalg.norm(data, axis = 1)
                data = data/np.max(data)
                self.plot(data, model_name, "alpaca", element_name="mlp")

            with open(f"data/alpaca/{model_name}/activation_resid.pkl", "rb") as f:
                data = pickle.load(f)
                data = np.linalg.norm(data, axis = 1)
                data = data/np.max(data)
                self.plot(data, model_name, "alpaca", element_name="resid")
        
        for model_name in self.model_names:
            
            with open(f"data/summarisation/{model_name}/activation_mlp.pkl", "rb") as f:
                data = pickle.load(f)
                data = np.linalg.norm(data, axis = 1)
                data = data/np.max(data)
                self.plot(data, model_name, "summarisation", element_name="mlp")
            
            with open(f"data/summarisation/{model_name}/activation_resid.pkl", "rb") as f:
                data = pickle.load(f)
                data = np.linalg.norm(data, axis = 1)
                data = data/np.max(data)
                self.plot(data, model_name, "summarisation", element_name="resid")
            
            with open(f"data/summarisation/{model_name}/activation_attn.pkl", "rb") as f:
                
                data = pickle.load(f)
                data = np.linalg.norm(data, axis = 1)
                data = data/np.max(data)
                self.plot(data, model_name, "summarisation", element_name="attn")
    
    def plot(self, data, model_name, dataset_name, element_name):
        
        y_values = data
        sizes = [int(y * 10)*10 for y in y_values]  # Make size proportional to the y-values
        x_values = ["layer " + str(i) for i in range(len(data))]

        # Create scatter plot
        fig = go.Figure(data=go.Scatter(
            x=x_values,
            y=y_values,
            mode='markers',
            marker=dict(
                size=sizes,  # Circle size is proportional to y-values
                color=y_values,  # Optional: color based on y-values
                showscale=True,  # Show color scale on the side
                colorscale='Viridis'  # Color scheme
            )
        ))

        # Customize the layout
        fig.update_layout(
            title=f"[{model_name}-{dataset_name}] Activation Analysis",
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            showlegend=False
        )

        fig.write_image(f"figures/{dataset_name}/{model_name}/activation_analysis_{element_name}.png")

if __name__ == "__main__":
    
    act_analysis = act_analysis_graph()
    act_analysis.forward()