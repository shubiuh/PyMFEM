# Here's the workflow:

---

## Using your own MFEM branch

### Step 1 — Fork and clone your MFEM branch

The default MFEM source is mfem. Since it was cloned by PyMFEM's build system, you can just set a remote and switch branches:

```bash
cd /workspace/PyMFEM/external/mfem
git remote set-url origin https://github.com/YOUR_USERNAME/mfem.git
# or keep upstream as origin and add your fork:
git remote add myfork https://github.com/YOUR_USERNAME/mfem.git
git checkout -b my-feature-branch
```

Or clone your fork to a different location:

```bash
git clone https://github.com/YOUR_USERNAME/mfem.git /workspace/my-mfem
cd /workspace/my-mfem
git checkout my-feature-branch
```

### Step 2 — Build only MFEM (after source changes)

```bash
# Rebuild in-place (fastest, no reinstall of PyMFEM needed if only .cpp changed)
cd /workspace/PyMFEM/external/mfem/cmbuild_par
make -j$(nproc) && cmake --install .
```

If you used a custom source location, reconfigure first:
```bash
cmake -DCMAKE_CXX_COMPILER=mpic++ -DMFEM_USE_MPI=1 \
      -DCMAKE_INSTALL_PREFIX=... /workspace/my-mfem
make -j$(nproc) && cmake --install .
```

### Step 3 — Rebuild PyMFEM Python wrappers

**Only `.cpp` implementation changed** (no header changes):
```bash
# No rebuild needed — Python loads libmfem.so dynamically
# Just reinstall if the .so changed:
pip install . -C"with-parallel=Yes" -C"skip-ext=Yes" -C"skip-swig=Yes"
```

**Headers changed** (new classes/methods):
```bash
# Point to your custom source for SWIG
pip install . -C"with-parallel=Yes" -C"skip-ext=Yes" \
  -C"mfem-source=/workspace/my-mfem" -C"swig=Yes"
# Then compile the wrappers:
pip install . -C"with-parallel=Yes" -C"skip-ext=Yes" -C"skip-swig=Yes"
```

### Step 4 — Push your MFEM changes

```bash
cd /workspace/PyMFEM/external/mfem   # or /workspace/my-mfem
git add -p                           # stage your changes
git commit -m "my change"
git push myfork my-feature-branch
```

---

**Key option summary:**

| Option | Purpose |
|---|---|
| `-C"mfem-source=<path>"` | Use a custom MFEM source directory |
| `-C"mfem-branch=<branch>"` | Checkout a specific branch when cloning |
| `-C"skip-ext=Yes"` | Skip rebuilding hypre/metis/mfem C++ |
| `-C"skip-swig=Yes"` | Skip SWIG wrapper regeneration |
| `-C"swig=Yes"` | Only regenerate SWIG wrappers, then exit |