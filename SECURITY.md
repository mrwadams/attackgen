# Security Policy

## Supported Versions

We release security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| latest (main branch)  | :white_check_mark: |

Always use the latest version of AttackGen for the best security.

## Reporting a Vulnerability

We take the security of AttackGen seriously. If you discover a security vulnerability, please follow these steps:

1. **Do Not** open a public GitHub issue for the vulnerability.
2. Email your findings to the project maintainers via GitHub (use the Security tab to report privately).
3. Provide detailed information including:
   - Description of the vulnerability
   - Steps to reproduce the issue
   - Potential impact
   - Suggested fix (if available)

We will acknowledge receipt of your vulnerability report within 48 hours and will send a more detailed response within 5 business days indicating the next steps in handling your report.

## Security Best Practices

### API Key Management

AttackGen requires API keys for various LLM providers. Follow these practices to keep your credentials secure:

1. **Never commit API keys to version control**
   - Use `.env` files for local development (already in `.gitignore`)
   - Use environment variables or secrets management in production

2. **Rotate API keys regularly**
   - Change keys periodically and after any suspected compromise
   - Use separate keys for development and production environments

3. **Limit API key permissions**
   - Use provider-specific settings to restrict key capabilities where possible
   - Monitor API usage for anomalies

4. **Secure secrets in deployment**
   - Use Streamlit Cloud secrets management for cloud deployments
   - Use Docker secrets or environment variables for containerized deployments
   - Never expose `.env` or `secrets.toml` files publicly

### Deployment Security

When deploying AttackGen:

1. **Network Security**
   - Deploy behind authentication/authorization if hosting publicly
   - Use HTTPS/TLS for all network traffic
   - Consider using a reverse proxy (nginx, Caddy) with security headers

2. **Docker Security**
   - Run containers with non-root users
   - Scan Docker images for vulnerabilities regularly
   - Keep base images updated

3. **Access Control**
   - Implement authentication for production deployments
   - Use Streamlit's built-in authentication features or external auth providers
   - Restrict access to authorized users only

4. **Data Privacy**
   - Be aware that generated scenarios may be sent to third-party LLM providers
   - Review provider privacy policies and data handling practices
   - Consider using local models (Ollama) for sensitive environments
   - Avoid including sensitive organizational details in scenario generation

### Secure Development

1. **Dependency Management**
   - Regularly update dependencies to patch security vulnerabilities
   - Review `requirements.txt` for outdated or vulnerable packages
   - Automated scanning with `pip-audit` and `safety` runs on every commit (see Automated Security Scanning section)

2. **Input Validation**
   - The application validates inputs, but always review user-provided data
   - Be cautious with custom API endpoints and credentials

3. **Code Reviews**
   - Review pull requests for security implications
   - Pay special attention to changes in authentication, API key handling, or data processing

## Automated Security Scanning

AttackGen uses comprehensive automated security scanning in our CI/CD pipeline to identify and address vulnerabilities proactively.

### Security Scanning Tools

Our security workflow runs automatically on every push and pull request, using:

1. **Python Code Analysis**
   - **Bandit**: Identifies common security issues in Python code (SQL injection, hardcoded credentials, etc.)
   - **CodeQL**: GitHub's advanced semantic analysis for security vulnerabilities

2. **Dependency Scanning**
   - **pip-audit**: Scans Python dependencies against the OSV vulnerability database
   - **Safety**: Checks dependencies using PyUp's comprehensive vulnerability database
   - **Dependabot**: Automated dependency updates and security alerts

3. **Secret Detection**
   - **Gitleaks**: Scans git history for accidentally committed secrets, API keys, and credentials

4. **Container Security**
   - **Trivy**: Scans Docker images for vulnerabilities in OS packages and dependencies

5. **Dependency Review**
   - Automated review of dependency changes in pull requests
   - Blocks dependencies with moderate or higher severity vulnerabilities
   - Prevents introduction of incompatible licenses (GPL-3.0, AGPL-3.0)

### Viewing Security Results

Security scan results are available in multiple locations:

- **GitHub Security Tab**: View all code scanning alerts, secret scanning alerts, and Dependabot alerts
- **Pull Request Checks**: Security scans run on every PR with results visible in the checks
- **Actions Tab**: Detailed workflow logs and security reports
- **Workflow Artifacts**: Downloadable JSON reports (Bandit, pip-audit) for deeper analysis

### Continuous Monitoring

- **Scheduled Scans**: Security workflow runs weekly (Mondays at 9 AM UTC) to catch newly disclosed vulnerabilities
- **Push-based Scans**: Automatic scanning on every push to `main` and `develop` branches
- **Pull Request Scanning**: All PRs are scanned before merge

### Addressing Security Findings

When security issues are identified:

1. **Critical/High Severity**: Addressed immediately with patches
2. **Medium Severity**: Evaluated and fixed in the next release
3. **Low Severity**: Reviewed and addressed based on impact and exploitability
4. **False Positives**: Documented and suppressed with justification

Developers can run security scans locally:
```bash
# Install security tools
pip install bandit pip-audit

# Run Bandit
bandit -r . -f txt

# Run pip-audit
pip-audit --desc
```

## Responsible Use Policy

AttackGen is designed for **legitimate cybersecurity testing and training purposes only**. Users should:

1. **Only use on authorized systems**
   - Obtain proper authorization before testing any systems
   - Respect organizational policies and legal requirements

2. **Follow ethical guidelines**
   - Use generated scenarios responsibly
   - Do not use for malicious purposes or unauthorized penetration testing

3. **Comply with applicable laws**
   - Follow all local, state, and federal laws regarding computer security
   - Respect the Computer Fraud and Abuse Act (CFAA) and similar legislation

## Known Security Considerations

1. **LLM Provider Data Sharing**
   - Generated scenarios and inputs are sent to third-party LLM providers
   - These providers may log requests for their own purposes
   - Review each provider's data handling policies

2. **Streamlit Session State**
   - Session data is stored in browser memory
   - Clear sensitive information when finished

3. **MITRE ATT&CK Data**
   - The application uses publicly available threat intelligence
   - Generated scenarios are educational and should be validated before use

4. **Rate Limiting**
   - Implement rate limiting if exposing the application publicly
   - Monitor API usage to prevent abuse

## Security Updates

Security updates will be released as patch versions and documented in the release notes. Users are encouraged to:

- Watch the repository for security advisories
- Subscribe to release notifications
- Keep installations up to date

## Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Streamlit Security Documentation](https://docs.streamlit.io/)
- [Docker Security Best Practices](https://docs.docker.com/engine/security/)
- [MITRE ATT&CK Framework](https://attack.mitre.org/)

## Acknowledgments

We appreciate the security research community's efforts in responsibly disclosing vulnerabilities. Contributors who report valid security issues will be acknowledged (with permission) in our security advisories.
