for file in "$@"
do
    n=$(grep -oE '"id": "[^ ]*"'  "$file" | uniq | wc -l)
    echo "$file:"
    echo "  $n unique ids"
done
