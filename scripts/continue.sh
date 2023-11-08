# Continue soln/test generation jobs
prefix="analyze"
regex='"id": "(\S+):([0-9]+)"'
root="/home/djl328/prob-repl"

for file in $root/$prefix-*.out
do
    echo $file
    line=$(tail -1 $file)
    if [[ $line =~ $regex ]]
    then
        name="${BASH_REMATCH[1]}"
        num=$(("${BASH_REMATCH[2]}" - 1))
        echo "Continuing run for ${name} from id ${num}"
	sbatch --requeue $root/slurm/solve.sub "$name" "$num"
    else
        echo "$file doesn't have a valid last id" >&2
    fi
done
