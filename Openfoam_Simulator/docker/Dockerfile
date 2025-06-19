FROM ubuntu:22.04

LABEL maintainer="F3D_visualizer Team"
LABEL description="Oil & Gas CFD Visualization System with PyQt, ParaView and OpenFOAM"

# Avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    git \
    curl \
    unzip \
    nano \
    build-essential \
    cmake \
    python3 \
    python3-pip \
    python3-dev \
    vim \
    ssh \
    sudo \
    locales \
    && rm -rf /var/lib/apt/lists/*

# Set up locale
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Install OpenFOAM-12 repository
RUN sh -c "wget -O - https://dl.openfoam.org/gpg.key > /etc/apt/trusted.gpg.d/openfoam.asc" \
    && add-apt-repository http://dl.openfoam.org/ubuntu

# Install OpenFOAM-12
RUN apt-get update && apt-get install -y \
    openfoam12 \
    && rm -rf /var/lib/apt/lists/*

# Add ParaView repository
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null \
    && apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" 

# Install ParaView and GUI dependencies
RUN apt-get update && apt-get install -y \
    paraview \
    python3-pyqt5 \
    python3-pyqt5.qtsvg \
    python3-numpy \
    python3-scipy \
    python3-matplotlib \
    python3-pandas \
    libgl1-mesa-glx \
    libglu1-mesa \
    xvfb \
    x11vnc \
    websockify \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Some packages might not be available, install them with pip
RUN pip3 install PyQtWebEngine

# Set up noVNC manually since the package might not be available
RUN mkdir -p /usr/local/novnc \
    && curl -k -L -o /tmp/novnc.zip https://github.com/novnc/noVNC/archive/v1.3.0.zip \
    && unzip /tmp/novnc.zip -d /usr/local/novnc/ \
    && mv /usr/local/novnc/noVNC-1.3.0/* /usr/local/novnc/ \
    && curl -k -L -o /tmp/websockify.zip https://github.com/novnc/websockify/archive/v0.10.0.zip \
    && unzip /tmp/websockify.zip -d /usr/local/novnc/ \
    && mv /usr/local/novnc/websockify-0.10.0 /usr/local/novnc/utils/websockify \
    && rm -f /tmp/novnc.zip /tmp/websockify.zip

# Add OpenFOAM environment setup to .bashrc
RUN echo "source /opt/openfoam12/etc/bashrc" >> /root/.bashrc

# Install Python packages directly without using requirements.txt
# Using --no-cache-dir to avoid hash verification issues
RUN pip3 install --upgrade pip --no-cache-dir && \
    # Core dependencies
    pip3 install numpy scipy pandas matplotlib PyYAML --no-cache-dir && \
    # PyQt dependencies
    pip3 install PyQt5 PyQt5-sip PyQtWebEngine --no-cache-dir && \
    # VTK and visualization
    pip3 install vtk pyqtgraph plotly seaborn --no-cache-dir && \
    # Data processing
    pip3 install h5py dask xarray --no-cache-dir && \
    # OpenFOAM Python interfaces
    pip3 install PyFoam --no-cache-dir && \
    # Mesh handling
    pip3 install meshio pyvista --no-cache-dir && \
    # Utilities
    pip3 install tqdm colorlog joblib rich --no-cache-dir && \
    # Testing
    pip3 install pytest pytest-qt --no-cache-dir

# Create application directory structure
RUN mkdir -p /app/F3D_visualizer/src /app/F3D_visualizer/data /app/F3D_visualizer/runs

# Copy supervisord config
COPY docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Copy entrypoint script
COPY docker/entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh

# Create minimal structure for initial run
RUN mkdir -p /app/F3D_visualizer/src/gui /app/F3D_visualizer/src/utils
RUN echo "class MainWindow: pass" > /app/F3D_visualizer/src/gui/__init__.py
RUN echo "class MainWindow: pass" > /app/F3D_visualizer/src/gui/main_window.py
RUN echo "def setup_logger(): pass" > /app/F3D_visualizer/src/utils/__init__.py
RUN echo "def setup_logger(): pass" > /app/F3D_visualizer/src/utils/logger.py
RUN echo "def load_config(): return {}" > /app/F3D_visualizer/src/config.py
RUN echo "print('F3D_visualizer starting...')" > /app/F3D_visualizer/src/main.py

# Create OpenFOAM run directory with correct permissions
RUN chmod -R 777 /app/F3D_visualizer/runs

# Expose ports for VNC and web access
EXPOSE 5900 6080

# Set entrypoint
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["shell"]