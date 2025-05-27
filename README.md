# 🧠 DAZL Graphical Language Compiler

**DAZL** is a custom-built domain-specific graphical programming language designed for educational visualization and intuitive algorithm expression. This compiler allows users to write high-level pseudo-code with visual instructions, which is then compiled into intermediate representation (IR), optimized, converted to assembly-like code, and finally rendered as visual output (shapes, charts, and 3D objects).

## 🌟 Features

* 📌 **Custom Language Syntax** for easy-to-write instructions (e.g., `CIRCLE x y R`)
* ⚙️ **Compiler Pipeline**:

  * Lexical Analysis
  * Parsing (AST)
  * Semantic Analysis
  * Intermediate Representation (IR) Generation
  * IR Optimization
  * Assembly Code Generation
  * Machine Code Simulation (Shape and Chart Rendering)
    
* 📊 **Graphical Visualization**:

  * 2D Shapes: Circle, Square, Rectangle, Triangle, etc.
  * 3D Shapes: Cube, Sphere, Cylinder, Cone, etc.
  * Charts: Bar, Pie, Line, Histogram, Scatter, etc.
* 🖥️ **Streamlit Web UI** for live code execution and rendering
* ✅ **Error Handling** for syntax and runtime issues
* 🧠 **Live Preview** and **Animated UI**

---

## 🛠️ How It Works

1. **Write DAZL Code** using high-level visual commands:

   ```dazl
   CIRCLE 10 10 10
   SHOW
   ```

2. **Compile** using the Streamlit app interface.

3. **Backend Flow**:

   * Tokenize and Parse DAZL code
   * Perform semantic checks
   * Generate IR and optimize
   * Generate and execute simulated machine code
   * Render output using `matplotlib`

---

## 🖥️ Run the Application

### 🔧 Prerequisites

* Python 3.8+
* Pip

### 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

> If `requirements.txt` is missing, install manually:

```bash
pip install streamlit matplotlib numpy pandas mplfinance networkx
```

### 🚀 Launch App

```bash
streamlit run app.py
```

> This will open a web browser interface where you can input DAZL code and visualize output.

---

## 🧪 Example DAZL Code

```dazl
SET R = 10
CIRCLE 20 20 R
SHOW
```

```dazl
SET data = [10, 20, 30]
BARCHART data
SHOW
```

---

## 📁 Project Structure

```
.
├── app.py               # Streamlit frontend
├── Dazl_Compiler.py     # Full compiler backend
├── xyz.dazl             # Example source code (DAZL format)
├── README.md            # This file
```

---

## 📌 Supported Keywords

* **Shapes**: `CIRCLE`, `SQUARE`, `RECTANGLE`, `TRIANGLE`, `OVAL`, `CUBE`, `SPHERE`, `CYLINDER`, `PYRAMID`, `SHOW`
* **Charts**: `BARCHART`, `PIECHART`, `LINECHART`, `HISTOGRAM`, `SCATTERPLOT`
* **Flow Control**: `IF`, `ELSE`, `FOR`, `WHILE`
* **Data Structures**: `LIST`, `APPEND`, `REMOVE`, `LENGTH`, `DISPLAY`

---

## 📚 Credits

Built using:

* 🐍 Python
* 📊 Matplotlib, NumPy, Pandas
* 🌐 Streamlit
* 🔍 Custom-built compiler and parser

---

## 📝 License

This project is for academic and educational use. Please credit the original author if reused.

