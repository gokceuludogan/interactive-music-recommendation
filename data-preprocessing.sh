awk -F $'\t' '{print $5}' userid-timestamp-artid-artname-traid-traname.tsv | sort -n | uniq | wc -l 
# number of uniq trackid: 961417

awk -F $'\t' '{ print $4 ";" $5 ";"$6}' userid-timestamp-artid-artname-traid-traname.tsv | sort -n | uniq -c > lastfm_uniq_songs.csv

awk ' $1 > 9 {print}' lastfm_uniq_songs.csv > lastfm_uniq_songs_ge9.csv

cut -c 9- lastfm_uniq_songs_ge9.csv > lastfm_uniq_songs_ge9_wout_c.csv 

awk -F\; 'NF==3 {print}' lastfm_uniq_songs_ge9_wout_c.csv > lastfm_uniq_songs_ge9_wout_c_clean.csv

# Open the file with vi
# vi lastfm_uniq_songs_ge9_wout_c_clean.csv
# Remove quotes 
# :%s/"//g
# Write file
# :w lastfm_uniq_songs_ge9_wout_c_clean_wout_quotes.csv

