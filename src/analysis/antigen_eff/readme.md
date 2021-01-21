# exp1 - what happens if antigen is always there?
set r_chronic=0
vary ag dose and check readouts

# exp2 - what happens if antigen is there and we add chr cells?
set r_chronic = const
vary ag dose and check readouts - could go into same plot as no chronic scenario

# exp3 - what happens if chronic cells exert feedback?
add fb from chronic cells to r_prolif and r_chronic

# exp4 - what happens for different antigen dynamics?
change virus dynamics from spiky to flat