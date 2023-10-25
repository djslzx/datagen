n=$(grep -oE '"id": "[^ ]*"'  "$1" | uniq | wc -l)
echo "$n unique ids found"
