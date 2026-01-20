#!/bin/bash
# Setup script for benchmarking

# System updates
sudo apt update && sudo apt upgrade -y
sudo apt dist-upgrade -y
sudo apt autoremove -y
sudo apt autoclean

# Essential build tools and libraries
sudo apt install -y build-essential curl wget git vim nano
sudo apt install -y software-properties-common apt-transport-https ca-certificates gnupg lsb-release
sudo apt install -y gcc g++ make cmake pkg-config libssl-dev
sudo apt install -y python3 python3-pip python3-dev python3-plotly
sudo apt install -y zip unzip tar gzip bzip2 xz-utils
sudo apt install -y htop neofetch tree ncdu
sudo apt install -y net-tools openssh-server ufw fail2ban

# Rust installation
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
rustup default stable
rustup component add rustfmt clippy rust-src

# Install bottom (system monitoring tool)
curl -LO https://github.com/ClementTsang/bottom/releases/download/0.11.2/bottom_0.11.2-1_amd64.deb
sudo dpkg -i bottom_0.11.2-1_amd64.deb
rm bottom_0.11.2-1_amd64.deb

# Install zellij (terminal workspace/multiplexer)
cargo install --locked zellij

# Set up firewall (basic)
sudo ufw allow OpenSSH
sudo ufw --force enable

# Configure fail2ban
sudo systemctl enable fail2ban
sudo systemctl start fail2ban

# Set timezone
sudo timedatectl set-timezone UTC

# Enable unattended security updates
sudo apt install -y unattended-upgrades
sudo dpkg-reconfigure -plow unattended-upgrades
