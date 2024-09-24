from imports import *

class act_analysis_graph_1:
    
    def __init__(self):
        self.model_names = ["Pythia14m", "Pythia70m", "Pythia160m", "Pythia410m", "Pythia1b", "Pythia1.4b"]
    
    def forward(self):
        
        for model_name in self.model_names:
            
            with open(f"data/tinystories/{model_name}/activation_attn.pkl", "rb") as f:
                data = pickle.load(f)
                data = np.linalg.norm(np.exp(data), axis = 1)
                self.plot(np.log(data), model_name, "tinystories", element_name="attn")
            
            with open(f"data/tinystories/{model_name}/activation_mlp.pkl", "rb") as f:
                data = pickle.load(f)
                data = np.linalg.norm(np.exp(data), axis = 1)
                self.plot(np.log(data), model_name, "tinystories", element_name="mlp")
            
            with open(f"data/tinystories/{model_name}/activation_resid.pkl", "rb") as f:
                data = pickle.load(f)
                data = np.linalg.norm(np.exp(data), axis = 1)
                self.plot(np.log(data), model_name, "tinystories", element_name="resid")
            
        for model_name in self.model_names:
            
            with open(f"data/alpaca/{model_name}/activation_attn.pkl", "rb") as f:
                data = pickle.load(f)
                data = np.linalg.norm(np.exp(data), axis = 1)
                self.plot(np.log(data), model_name, "alpaca", element_name="attn")
            
            with open(f"data/alpaca/{model_name}/activation_mlp.pkl", "rb") as f:
                data = pickle.load(f)
                data = np.linalg.norm(np.exp(data), axis = 1)
                self.plot(np.log(data), model_name, "alpaca", element_name="mlp")

            with open(f"data/alpaca/{model_name}/activation_resid.pkl", "rb") as f:
                data = pickle.load(f)
                data = np.linalg.norm(np.exp(data), axis = 1)
                self.plot(np.log(data), model_name, "alpaca", element_name="resid")
        
        for model_name in self.model_names:
            
            with open(f"data/summarisation/{model_name}/activation_mlp.pkl", "rb") as f:
                data = pickle.load(f)
                data = np.linalg.norm(np.exp(data), axis = 1)
                self.plot(np.log(data), model_name, "summarisation", element_name="mlp")
            
            with open(f"data/summarisation/{model_name}/activation_resid.pkl", "rb") as f:
                data = pickle.load(f)
                data = np.linalg.norm(np.exp(data), axis = 1)
                self.plot(np.log(data), model_name, "summarisation", element_name="resid")
            
            with open(f"data/summarisation/{model_name}/activation_attn.pkl", "rb") as f:
                
                data = pickle.load(f)
                data = np.linalg.norm(np.exp(data), axis = 1)
                self.plot(np.log(data), model_name, "summarisation", element_name="attn")
    
    def plot(self, data, model_name, dataset_name, element_name):
            
        y_values = data
        x_values = ["layer " + str(i) for i in range(len(data))]

        # Create scatter plot with lines connecting the points
        fig = go.Figure(data=go.Scatter(
            x=x_values,
            y=y_values,
            mode='lines+markers',  # Connect the dots with lines and show markers
            marker=dict(
                color='blue',  # Set all markers to the same color (e.g., blue)
                size=10  # Adjust the marker size if needed
            )
        ))

        # Customize the layout
        fig.update_layout(
            title=f"[{model_name}-{dataset_name}] Activation Analysis",
            xaxis_title='Layers',
            yaxis_title='Activation Value (Log-Scale)',
            showlegend=False
        )

        fig.write_image(f"figures/{dataset_name}/{model_name}/activation_analysis_{element_name}.png")



class act_analysis_graph_2:
    
    def __init__(self):
        self.model_names = ["Pythia14m", "Pythia70m", "Pythia160m", "Pythia410m", "Pythia1b", "Pythia1.4b"]
    
    def forward(self):
        
        for dataset in [["tinystories", "145 words/string"], ["alpaca", "7 words/string"], ["summarisation", "340 words/string"]]:
            final_data_attn = {}
            print(dataset)
            for model_name in self.model_names:
                
                with open(f"data/{dataset[0]}/{model_name}/activation_attn.pkl", "rb") as f:
                    data = pickle.load(f)
                
                data = np.linalg.norm(np.exp(data), axis = 1)
                final_data_attn[model_name] = np.log(data)
            
            
            self.plot(
                data = final_data_attn, 
                dataset_name = dataset[0],
                words = dataset[1], 
                element_name = "attn", 
                models_names = self.model_names
                )

            final_data_mlp = {}
            
            for model_name in self.model_names:
                
                with open(f"data/{dataset[0]}/{model_name}/activation_mlp.pkl", "rb") as f:
                    data = pickle.load(f)
                
                data = np.linalg.norm(np.exp(data), axis = 1)
                final_data_mlp[model_name] = np.log(data)
            
            
            self.plot(
                data = final_data_mlp, 
                dataset_name = dataset[0],
                words = dataset[1], 
                element_name = "mlp", 
                models_names = self.model_names
                )
            
            
            final_data_resid = {}
            
            for model_name in self.model_names:
                
                with open(f"data/{dataset[0]}/{model_name}/activation_resid.pkl", "rb") as f:
                    data = pickle.load(f)
                
                data = np.linalg.norm(np.exp(data), axis = 1)
                final_data_resid[model_name] = np.log(data)
            
            
            self.plot(
                data = final_data_resid, 
                dataset_name = dataset[0],
                words = dataset[1], 
                element_name = "resid", 
                models_names = self.model_names
                )
        
        
            

    def plot(self, data, dataset_name, element_name, models_names, words):
        # Create an empty figure
        fig = go.Figure()

        # Generate a list of colors ranging from light to dark blue
        cmap = matplotlib.cm.get_cmap('winter')  # Use 'winter' colormap (greenish-blue to blue)
        colors = [matplotlib.colors.rgb2hex(cmap(0.1 + 0.8 * (i / (len(models_names) - 1)))) for i in range(len(models_names))]
        colors.reverse()
            # Reverse the order of the colors

        # Set the last color to black
        if colors:
            colors[-1] = '#000000'  # Black color

        # Loop over each model and add a trace for its data
        for idx, model_name in enumerate(models_names):
            y_values = data[model_name]  # Get the activation values for this model
            x_values = [(i+1)/len(y_values) for i in range(len(y_values))]  # Corresponding x-values

            # Add a scatter trace for this model's data, with progressively darker blue color
            fig.add_trace(go.Scatter(
                x=x_values,
                y=y_values,
                mode='lines+markers',  # Connect the dots with lines and show markers
                name=model_name,  # Set the legend label to the model name
                marker=dict(
                    color=colors[idx],  # Set color for each model
                    size=10  # Adjust the marker size if needed
                ),
                line=dict(color=colors[idx])  # Set line color
            ))

        # Customize the layout for the entire plot
        fig.update_layout(
            title=f"[{dataset_name} {words}] Attention Activation Analysis for All Models",
            xaxis_title='Normalized Layers',
            yaxis_title='Activation Value Norm per layer (Log-Scale)',
            showlegend=True  # Show the legend to distinguish between models
        )

        # Save the figure for all models
        fig.write_image(f"figures/{dataset_name}/activation_analysis_{element_name}_all_models.png")



if __name__ == "__main__":
    
    act_analysis = act_analysis_graph_1()
    act_analysis.forward()
    
    act_analysis_ = act_analysis_graph_2()
    act_analysis_.forward()