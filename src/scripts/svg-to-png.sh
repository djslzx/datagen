HEIGHT=300
for file in "$1"/*.svg; do
  name="${file%.*}"
  echo $name
  rsvg-convert -h $HEIGHT "$file" > "$name.png"
done