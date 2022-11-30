Let's make a video of black holes orbiting and merging, 
starting from their EOB description!

Remember to 
```
export PYTHONPATH=$PYTHONPATH:~/path/to/teobresums/Python
```

Get the correct version of `TEOBResumS`: 

```bash
git clone https://bitbucket.org/eob_ihes/teobresums
cd teobresums
git checkout eccentric.v0_a6c_c3_circularized
git cherry-pick e8cb333
cd Python
make
```

because a function we need is not in the eccentric model they used
in the paper.