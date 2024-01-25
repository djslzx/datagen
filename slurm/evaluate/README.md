# Evaluation notes

The main slurm script is in `eval-arr.sub`.  Here's how it works.

## Configuration
The main thing to add to a typical slurm script config is the following line:
```
#SBATCH --array=1-500
```
This sets up jobs with ids in [1, 500].  More precisely, the `.sub` file describes a job array, where each element of the array is called a task.  Each task runs whatever follows the initialization step, i.e., the `#SBATCH ...` comments.

## Getting different tasks to do different things
Each task gets assigned a unique ID from our earlier array config line.  In this example, we would get task IDs in [1, 500].  In the script executed by each task, we can access the task ID via the variable `$SLURM_ARRAY_TASK_ID`, then use a configuration file that assigns values to each task.

For instance, if we want to run a job array that runs a script `$script` on 100 different files, we would set
```
#SBATCH --array=1-100
```

and then, given a configuration file `$config`
```
ID  Filename
1   file1.txt
2   file2.txt
...
```

we can fetch the appropriate file using an `awk` command
```
file=$(awk -v ID=$SLURM_ARRAY_TASK_ID '$1==ID {print $2}' $config)
```

and run the script:
```
$script --file $file
```