# Scam AI Flask Backend


## CI/CD Setup
### GitHub Configuration

#### Required Variables and Secrets

This project requires both GitHub Variables and Secrets for deployment configuration.

#### Variables
Configure these in Settings > Secrets and variables > Actions > Variables:

- `EC2_USERNAME`: The username for SSH access to EC2 instance (e.g., 'ubuntu', 'ec2-user')
- `EC2_HOST`: The public DNS or IP of your EC2 instance

##### Secrets
Configure these in Settings > Secrets and variables > Actions > Secrets:

- `EC2_SSH_KEY`: The private SSH key for EC2 instance access
  - This is specifically for the CI/CD workflow to connect to EC2
  - This is different from the SSH key used for GitHub-to-EC2 repository access
- `GIT_SSH_KEY`: The SSH key for GitHub repository access on EC2
  - This allows the EC2 instance to pull from private GitHub repositories
  - Generate a dedicated deploy key for this purpose

#### How to Configure

1. Navigate to your GitHub repository
2. Go to Settings > Secrets and variables > Actions
3. For Variables:
   - Click "Variables" tab
   - Click "New repository variable"
   - Add `EC2_USERNAME` and `EC2_HOST`
4. For Secrets:
   - Click "Secrets" tab
   - Click "New repository secret"
   - Add `EC2_SSH_KEY` and `GIT_SSH_KEY`

#### Note on Usage
- Variables (`EC2_USERNAME`, `EC2_HOST`) are visible in repository settings and can be reused across workflows
- Secrets (`EC2_SSH_KEY`, `GIT_SSH_KEY`) are encrypted and hidden for security purposes

### EC2 Setup

#### Required SSH Key
When launching a new EC2 instance, you MUST use:
- `scamai-detect` SSH key (CRITICAL - this specific key is required for deployment)

#### Required IAM Role
- `scamai-ec2` (for S3 access)

#### Steps for EC2 Launch
1. During EC2 instance creation, select `scamai-detect` as the key pair
   - This specific SSH key is essential for deployment processes
2. Under "IAM instance profile", select `scamai-ec2`
3. Continue with remaining EC2 configuration steps

> **Important**: Using the `scamai-detect` SSH key is mandatory. The deployment process is configured to use this specific key and will fail without it.