call cd ..
call conda env create -f environment.yml
call conda activate ElectricInsights
call ipython kernel install --user --name=ElectricInsights
pause