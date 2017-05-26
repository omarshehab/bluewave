import pip

installed_packages = pip.get_installed_distributions()

flat_installed_packages = [package.project_name for package in installed_packages]

needed_packages = ["xclip", "sympy", "opencv-python", "scikit-image", "pyquil", "sapi", "pymaxflow", "jupyter", "qutip", "tensorflow", "cycler",
 "scipy", "numpy", "pillow"]

for package in needed_packages:
   if package in flat_installed_packages:
      print package + " installed"
   else:
      print package + " not installed"

