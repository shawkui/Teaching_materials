## 1. Getting Started with Anaconda
----
### 1.1 Introduction to Anaconda

**What is Anaconda?**
Anaconda is a powerful distribution of Python and R designed specifically for scientific computing and data science. It comes packed with numerous libraries essential for data analysis, machine learning, and visualization, making it an ideal choice for researchers and developers alike.

**Key Components:**
- **Anaconda Navigator:** A user-friendly graphical interface that simplifies the management of environments and packages.
- **Conda:** A robust package manager that streamlines the installation and updating of libraries and their dependencies.
- **Included Software:** Pre-installed with a wide array of popular data science tools, saving you time and effort in setting up your development environment.
----
### 1.2 Installation of Anaconda

#### Downloading Anaconda
* Visit the official Anaconda download page at [https://www.anaconda.com/download/success](https://www.anaconda.com/download/success).
* Select the installer that corresponds to your operating system (Windows, macOS, or Linux).

#### Installing Anaconda on Windows

1. **Download the Installer**
   - Locate the latest version for Windows and download the Graphical Installer (.exe file).

2. **Run the Installer**
   - Double-click the downloaded .exe file to initiate the installation wizard.
   - Follow the on-screen prompts.
   - When asked, choose "Just Me" if this is a personal installation or "All Users" if you wish to make it available to everyone on the computer.
   - Ensure the checkbox "Add Anaconda3 to my PATH environment variable" is checked to facilitate easy access to Anaconda commands.

3. **Verify Installation**
   - Launch Anaconda Navigator from your Start menu and confirm it opens successfully.

Thank you for pointing that out. You're correct; the installer for macOS is indeed a `.pkg` file, not a `.sh` script. Let's correct that section to reflect the accurate installation process for macOS:

----
#### Installing Anaconda on macOS

1. **Download the Installer**
   - Select the latest version for macOS and download the Installer Package (.pkg file).

2. **Run the Installer**
   - Double-click the downloaded `.pkg` file to start the installation process.
   - Follow the on-screen prompts.
   - Accept the license agreement.
   - Use the default installation path unless you have a specific reason to change it.
   - Choose to prepend Anaconda to your PATH in either your `.bash_profile` or `.zshrc` file (for users of the Z shell).

3. **Verify Installation**
   - Find Anaconda Navigator in your Applications folder and verify it launches without issues.

----
#### Installing Anaconda on Linux

1. **Download the Installer**
   - Pick the latest version for Linux and download the Shell Script Installer (.sh file).

2. **Run the Installer**
   - Open Terminal.
   - Change directory to where the .sh file is located.
   - Make the installer executable:
     ```bash
     chmod +x Anaconda3-<version>-Linux-x86_64.sh
     ```
   - Execute the installer:
     ```bash
     ./Anaconda3-<version>-Linux-x86_64.sh
     ```
   - Adhere to the on-screen instructions.
   - Agree to the terms of use.
   - Use the default installation directory unless otherwise needed.
   - Decide whether to prepend Anaconda to your PATH in your shell configuration file (such as `.bashrc`).

3. **Verify Installation**
   - Restart your terminal session.
   - Type `conda list` to display a list of installed packages.
   - If installed correctly, Anaconda and Conda should appear in the list of packages.
----
#### Additional Steps

- **Environment Variables:**
  - On Windows, if you opted out of adding Anaconda to your PATH during installation, you can add it manually via System Properties.
  - On macOS and Linux, ensure your shell profile includes the correct path to Anaconda.

- **Testing Installation:**
  - Open Anaconda Navigator and start Jupyter Notebook to ensure everything is functioning as expected.

---
---

## 2. Jupyter Notebooks

### 2.1 Starting a Jupyter Notebook

To start a Jupyter Notebook, follow these steps based on your operating system:

#### For Windows (three ways):
- Click the **Windows Menu** -> **Anaconda 3** -> **Jupyter Notebook**.
- Open **Anaconda Navigator** and Click on “Launch” in the **Jupyter Notebook box**
- Open **Anaconda Prompt** and Type “jupyter notebook” (without quotes) and hit the return key

#### For macOS:
- Open a terminal window (found in the **Utilities** folder).
- Enter the command: `jupyter notebook`.
- Alternatively, to specify your preferred browser, use: `jupyter notebook --browser=firefox`.

You can also launch Jupyter Notebook through **Anaconda Navigator**, located in your **Applications** folder.

#### For Linux:
- Open a terminal.
- Type: `jupyter notebook`.
- To choose a specific browser, enter: `jupyter notebook --browser=firefox`.

Once Jupyter starts, your default web browser will open, displaying the Jupyter dashboard. To create a new notebook, click the **New** button located in the upper-right corner of the dashboard. Select **Python 3** or any other kernel you have installed. This action will open a new Jupyter Notebook where you can begin writing code and performing analyses.

### 2.2 Usage

#### 2.2.1 How to open a Notebook file
![Img](https://github.com/shawkui/Teaching_materials/blob/main/DDA4340_computational_finance/imgs/Tu1_0.png?raw=true)


#### 2.2.2 How to open a Notebook file
![Img](https://github.com/shawkui/Teaching_materials/blob/main/DDA4340_computational_finance/imgs/Tu1_1.png?raw=true)

---
#### 2.2.3 How to start writing a Jupyter Notebook
![Img](https://github.com/shawkui/Teaching_materials/blob/main/DDA4340_computational_finance/imgs/Tu1_2.png?raw=true)
![Img](https://github.com/shawkui/Teaching_materials/blob/main/DDA4340_computational_finance/imgs/Tu1_3.png?raw=true)

---
#### 2.2.4 How to start writing a Jupyter Notebook
![Img](https://github.com/shawkui/Teaching_materials/blob/main/DDA4340_computational_finance/imgs/Tu1_4.png?raw=true)

Or, you can install package by run pip beginning with ! in the notebook as below:

```python
! pip install numpy
```

---
---

## 3. Markdown

Markdown is a lightweight markup language that you can use to format text in a simple way. Here are some common elements and how to use them:

----
### 3.1 Headings
To create headings, you can use the `#` symbol followed by a space and your heading text. The number of `#` symbols denotes the level of the heading:
```markdown
# Heading 1
## Heading 2
### Heading 3
```

This will render as:
# Heading 1
## Heading 2
### Heading 3

----
### 3.2 Emphasis
For bold or italic text, use asterisks (`*`) or underscores (`_`):
```markdown
*Italic Text* _Also Italic_
**Bold Text** __Also Bold__
```

This will render as:
*Italic Text* _Also Italic_

**Bold Text** __Also Bold__

----
### 3.3 Lists
- **Unordered lists** (bullet points) use an asterisk (`*`), plus sign (`+`), or hyphen (`-`):
  ```markdown
  * Item 1
  * Item 2
  * Item 3
  ```
  This will render as:
  * Item 1
  * Item 2
  * Item 3

- **Ordered lists** (numbered) use numbers followed by a period (`.`):
  ```markdown
  1. First Item
  2. Second Item
  3. Third Item
  ```

  This will render as:
  1. First Item
  2. Second Item
  3. Third Item

----
### 3.4 Code Blocks
To insert code, you can either use backticks (\`) for inline code or three backticks for a block of code:

```

Here is inline code: `print("Hello, world!")`

And here is a code block:

```python
def hello_world():
    print("Hello, world!")
```
```

This will render as:

Here is inline code: `print("Hello, world!")`

And here is a code block:

```python
def hello_world():
    print("Hello, world!")
```
----
### 3.5 Images
To insert an image, use an exclamation mark (`!`), followed by square brackets containing the alt text and parentheses containing the URL or local path of the image:
```markdown
![Img](https://github.com/shawkui/Teaching_materials/blob/main/DDA4340_computational_finance/imgs/markdown.png?raw=true)
```

This will render as:
![Img](https://github.com/shawkui/Teaching_materials/blob/main/DDA4340_computational_finance/imgs/markdown.png?raw=true)

----
### 3.6 Math Environment
Markdown supports rendering mathematical expressions using LaTeX syntax within specific delimiters like `$` for inline math and `$$` for display math:
```markdown
Inline math: $x = {-b \pm \sqrt{b^2-4ac} \over 2a}$

Display math:
$$
E = mc^2
$$
```

This will render as:

Inline math: $x = {-b \pm \sqrt{b^2-4ac} \over 2a}$

Display math:
$$
E = mc^2
$$


* Commonly used math symbols can be found [here](https://tug.ctan.org/info/undergradmath/undergradmath.pdf)

---
---
