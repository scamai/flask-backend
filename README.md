# Scam AI Flask Backend

## Assumptions and Prerequisites
- EC2 instance has GitHub SSH key configured to pull repository updates
  - Note: This SSH key is separate from the one used in the CI workflow

## GitHub Configuration

### Required Variables and Secrets

This project requires both GitHub Variables and Secrets for deployment configuration.

#### Variables
Configure these in Settings > Secrets and variables > Actions > Variables:

- `EC2_USERNAME`: The username for SSH access to EC2 instance (e.g., 'ubuntu', 'ec2-user')
- `EC2_HOST`: The public DNS or IP of your EC2 instance

#### Secrets
Configure these in Settings > Secrets and variables > Actions > Secrets:

- `EC2_SSH_KEY`: The private SSH key for EC2 instance access
  - This is specifically for the CI/CD workflow to connect to EC2
  - This is different from the SSH key used for GitHub-to-EC2 repository access

### How to Configure

1. Navigate to your GitHub repository
2. Go to Settings > Secrets and variables > Actions
3. For Variables:
   - Click "Variables" tab
   - Click "New repository variable"
   - Add `EC2_USERNAME` and `EC2_HOST`
4. For Secrets:
   - Click "Secrets" tab
   - Click "New repository secret"
   - Add `EC2_SSH_KEY`

### Note on Usage
- Variables (`EC2_USERNAME`, `EC2_HOST`) are visible in repository settings and can be reused across workflows
- Secrets (`EC2_SSH_KEY`) are encrypted and hidden for security purposes

## IAM Instance Profile Setup

### Required Profile
- Profile Name: `scamai-ec2`
- Already configured with `AmazonS3FullAccess` policy

### Why S3 Access is Required
The EC2 instance needs S3 access to:
- Store and retrieve AI model training data

### Steps to Attach Profile to EC2
1. EC2 Dashboard > Select Instance
2. Actions > Security > Modify IAM role
3. Select 'scamai-ec2' from dropdown
4. Click 'Update IAM role'