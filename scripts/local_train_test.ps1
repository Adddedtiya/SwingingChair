# Enable error handling and verbose output
$ErrorActionPreference = "Stop"
$VerbosePreference = "Continue"

# Define the path to Python executable and the script
$pythonPath = "C:\Users\alipa\AppData\Local\Microsoft\WindowsApps\python.exe"
$scriptPath = "C:\Users\alipa\repos\VariationalBeta\training_vaegan.py"

# Define the arguments
$application_args = @(
    "--dataset_root C:\Users\alipa\repos\VariationalBeta\datasets\choco",
    "--name local_runtest",
    "--total_epochs 3",
    "--batch_size 4",
    "--load_threads 0",
    "--latent_size 256"
)

$argsString = $application_args -join " "

# Execute the script with the arguments

# Start the process
Start-Process -FilePath $pythonPath -ArgumentList $scriptPath, $argsString -NoNewWindow -Wait
