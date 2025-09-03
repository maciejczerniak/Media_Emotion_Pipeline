# 🤝 Contributing Guidelines

Thank you for considering a contribution to this project! To keep things clean and collaborative, please follow the steps below.

---

## 🔀 Workflow: How to Contribute

1. **Checkout from `dev`**
   - Always create new branches from the `dev` branch:

     ```bash
     git checkout dev
     git pull
     git checkout -b your-feature-branch
     ```

2. **Make your changes**
   - Write clean, tested, and documented code.
   - Follow the style guide below.

3. **Commit your changes**
   - Use clear commit messages (see below).

4. **Push your branch**

   ```bash
   git push origin your-feature-branch
    ```

5. **Create a Pull Request**
    - Alwasy target the `dev` branch.
    - Provide a clear description of your changes.
    - Link any related issues.
    - GitHub will automatically require:
      - ✅ A pull request
      - ✅ At least 1 approval
      - ✅ A review from a Code Owner (if applicable)

## 📦 Commit Message Guidelines

Please use clear and consistent commit titles. Good commit messages help others understand your changes quickly.

### ✅ Good Examples

- `Fix login redirect issue on Firefox`
- `Add unit tests for payment service`
- `Refactor user model for better readability`

### ❌ Bad Examples

- `Update`
- `Bugfix`
- `Misc changes`
- `temp`
- `Fix some stuff`
- etc.

### 🛠️ Optional Format (Recommended)

```md
<type>: <short description>
```

#### Common types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only changes
- `style`: Formatting, missing semi colons, etc.
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `test`: Adding or fixing tests
- `chore`: Other changes (e.g., build tools, CI)

##### Examples

- `feat: add user profile settings page`
- `fix: correct typo in error message`
- `chore: update dependencies`  

## 🔒 Branch Protection Rules

The following rules are active on main and dev:

- ✅ Pull request required before merging
- ✅ At least 1 approval is required
  
Do not push directly to `main` or `dev`.
