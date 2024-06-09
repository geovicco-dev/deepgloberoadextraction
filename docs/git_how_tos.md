To pull changes from the `main` branch without overwriting your existing changes in your current branch in Git, you can follow these steps:

1. **Stash Your Changes (Optional)**:
   - If you have changes in your current branch that you haven't committed yet, you might want to stash them to temporarily store them away. This step is optional but can help avoid conflicts. You can stash your changes by running:
     ```
     git stash
     ```

2. **Fetch Changes from Main**:
   - Fetch the latest changes from the `main` branch:
     ```
     git fetch origin main
     ```

3. **Merge Changes into Your Branch**:
   - Merge the changes from the `main` branch into your current branch. This will apply the changes from `main` without overwriting your existing changes:
     ```
     git merge origin/main
     ```

4. **Resolve Conflicts (if any)**:
   - If there are conflicts between your changes and the changes from `main`, Git will notify you about it. You'll need to resolve these conflicts manually by editing the affected files. After resolving conflicts, you'll need to mark them as resolved using:
     ```
     git add <conflicted_files>
     ```
     Then continue the merge process by running:
     ```
     git merge --continue
     ```

5. **Apply Stashed Changes (if stashed)**:
   - If you stashed changes in Step 1, you can apply them back now:
     ```
     git stash pop
     ```

6. **Review Changes**:
   - After merging changes from `main` and resolving conflicts (if any), review the changes using:
     ```
     git diff
     ```

7. **Commit Changes**:
   - If you're satisfied with the changes, commit them to your branch:
     ```
     git commit -m "Merge changes from main branch"
     ```

8. **Push Changes (if desired)**:
   - If you want to push your changes to the remote repository, you can do so with:
     ```
     git push origin <your_branch_name>
     ```

By following these steps, you can safely merge changes from the `main` branch into your current branch without overwriting your existing changes.