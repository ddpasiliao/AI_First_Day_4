import streamlit as st
import openai
import plotly.graph_objects as go

# Streamlit interface
st.title("MaxLoad: Load Optimization Assistant")
st.write("Optimize load arrangements for better space utilization.")

# API Key Input
api_key = st.text_input("Enter your OpenAI API Key", type="password")
if api_key:
    openai.api_key = api_key

# User inputs for vehicle selection
vehicle_type = st.selectbox("Select Vehicle Type", ["Truck", "Van", "Container"])
total_weight_limit = st.number_input("Total Weight Limit (kg)", min_value=0.0)

# User inputs for item details
st.write("Enter item details:")
item_name = st.text_input("Item Name")
item_weight = st.number_input("Item Weight (kg)", min_value=0.0)
item_dimensions = {
    "Length (cm)": st.number_input("Item Length (cm)", min_value=0.0),
    "Width (cm)": st.number_input("Item Width (cm)", min_value=0.0),
    "Height (cm)": st.number_input("Item Height (cm)", min_value=0.0)
}
item_fragile = st.selectbox("Is the item fragile?", ["Yes", "No"])
item_stackable = st.selectbox("Is the item stackable?", ["Yes", "No"])

# Collect all item details
if st.button("Add Item"):
    item_details = {
        "Name": item_name,
        "Weight (kg)": item_weight,
        "Dimensions (LxWxH) cm": f"{item_dimensions['Length (cm)']} x {item_dimensions['Width (cm)']} x {item_dimensions['Height (cm)']}",
        "Fragile": item_fragile,
        "Stackable": item_stackable
    }
    st.session_state.items = st.session_state.get('items', []) + [item_details]
    st.write("Item added:", item_details)

# Show added items
if 'items' in st.session_state:
    st.write("Current Items:")
    for i, item in enumerate(st.session_state.items, start=1):
        st.write(f"{i}. {item}")

# Run Load Optimization
if st.button("Optimize Load") and api_key:
    if st.session_state.get('items'):
        # Formulate prompt for GPT
        items_info = "\n".join([f"{item['Name']}: Weight {item['Weight (kg)']} kg, Dimensions {item['Dimensions (LxWxH) cm']}, Fragile {item['Fragile']}, Stackable {item['Stackable']}"
                                for item in st.session_state.items])
        prompt = (f"Optimize the loading arrangement for the following items in a {vehicle_type} with weight limit {total_weight_limit} kg:\n{items_info}.\n"
                  "Arrange them to maximize space utilization and minimize risk of damage. Provide a clear, organized loading plan.")

        # Call GPT for optimization
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=300
        )

        # Show GPT's load optimization suggestion
        load_plan = response.choices[0].text.strip()
        st.write("Optimized Load Plan:", load_plan)

        # Visualization (simplified example using Plotly for 3D load representation)
        fig = go.Figure(data=[go.Scatter3d(x=[0, 1], y=[0, 1], z=[0, 1], mode='markers')])
        fig.update_layout(scene=dict(xaxis_title='Width', yaxis_title='Depth', zaxis_title='Height'))
        st.plotly_chart(fig)

    else:
        st.error("Please add item details before optimizing load.")
elif not api_key:
    st.warning("Please enter your API key.")