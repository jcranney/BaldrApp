{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = { self, nixpkgs }: let 
    system = "x86_64-linux";
    pkgs = import nixpkgs { inherit system; };
  in {

    packages.x86_64-linux.default = pkgs.mkShell rec {
      packages = with pkgs; [
        glib
        libGL
        fontconfig
        libxkbcommon
        wayland
        dbus
        freetype
        nlohmann_json
      ];
      # QT_DEBUG_PLUGINS = 1;
      shellHook = ''
        zsh
      '';
      LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath packages;
    };

  };
}
