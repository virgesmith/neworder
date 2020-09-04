#!/bin/bash

# this is truly horrible, but sed doesnt like multiline strings and pydoc churns out garish html if used within python

output=docs/api_auto.md

echo "# API Reference" > $output
echo >> $output
echo "TODO reformat raw docstrings" >> $output
echo >> $output
echo "# \`neworder\` module" >> $output
echo >> $output
echo \`\`\` >> $output
python -m pydoc neworder >> $output
echo \`\`\` >> $output
echo  >> $output
echo "# \`neworder.mpi\` module" >> $output
echo >> $output
echo \`\`\` >> $output
python -m pydoc neworder.mpi >> $output
echo \`\`\` >> $output
echo  >> $output
echo "# \`neworder.time\` module" >> $output
echo >> $output
echo \`\`\` >> $output
python -m pydoc neworder.time >> $output
echo \`\`\` >> $output
echo  >> $output
echo "# \`neworder.stats\` module" >> $output
echo >> $output
echo \`\`\` >> $output
python -m pydoc neworder.stats >> $output
echo \`\`\` >> $output
echo  >> $output
echo "# \`neworder.dataframe\` module" >> $output
echo >> $output
echo \`\`\` >> $output
python -m pydoc neworder.dataframe >> $output
echo \`\`\` >> $output
