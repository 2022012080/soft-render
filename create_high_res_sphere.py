import math

def generate_high_res_sphere(radius=2.0, longitude_divisions=32, latitude_divisions=16):
    """
    Generate a high-resolution sphere with the specified number of divisions.
    
    Args:
        radius: Sphere radius
        longitude_divisions: Number of divisions around the equator (must be even)
        latitude_divisions: Number of divisions from pole to pole
    """
    
    # Ensure longitude_divisions is even for proper pole handling
    if longitude_divisions % 2 != 0:
        longitude_divisions += 1
    
    vertices = []
    texture_coords = []
    normals = []
    faces = []
    
    # Generate vertices, texture coordinates, and normals
    for lat_idx in range(latitude_divisions + 1):
        # Latitude angle from 0 to π
        lat_angle = math.pi * lat_idx / latitude_divisions
        
        for lon_idx in range(longitude_divisions + 1):
            # Longitude angle from 0 to 2π
            lon_angle = 2 * math.pi * lon_idx / longitude_divisions
            
            # Convert spherical coordinates to Cartesian
            x = radius * math.sin(lat_angle) * math.cos(lon_angle)
            y = radius * math.cos(lat_angle)
            z = radius * math.sin(lat_angle) * math.sin(lon_angle)
            
            # Add vertex
            vertices.append((x, y, z))
            
            # Add texture coordinates
            u = lon_idx / longitude_divisions
            v = lat_idx / latitude_divisions
            texture_coords.append((u, v))
            
            # Add normal (unit vector from center to vertex)
            length = math.sqrt(x*x + y*y + z*z)
            nx = x / length
            ny = y / length
            nz = z / length
            normals.append((nx, ny, nz))
    
    # Generate faces
    for lat_idx in range(latitude_divisions):
        for lon_idx in range(longitude_divisions):
            # Calculate vertex indices for current quad
            v0 = lat_idx * (longitude_divisions + 1) + lon_idx
            v1 = lat_idx * (longitude_divisions + 1) + lon_idx + 1
            v2 = (lat_idx + 1) * (longitude_divisions + 1) + lon_idx + 1
            v3 = (lat_idx + 1) * (longitude_divisions + 1) + lon_idx
            
            # Add two triangles to form the quad
            # First triangle
            faces.append((v0 + 1, v0 + 1, v0 + 1,  # vertex/texture/normal
                         v1 + 1, v1 + 1, v1 + 1,
                         v2 + 1, v2 + 1, v2 + 1))
            
            # Second triangle
            faces.append((v0 + 1, v0 + 1, v0 + 1,  # vertex/texture/normal
                         v2 + 1, v2 + 1, v2 + 1,
                         v3 + 1, v3 + 1, v3 + 1))
    
    return vertices, texture_coords, normals, faces

def write_obj_file(filename, vertices, texture_coords, normals, faces):
    """Write the sphere data to an OBJ file."""
    with open(filename, 'w') as f:
        f.write(f"# High-Resolution Sphere Model\n")
        f.write(f"# Radius: 2.0\n")
        f.write(f"# Longitude divisions: 64\n")
        f.write(f"# Latitude divisions: 32\n")
        f.write(f"# Total vertices: {len(vertices)}\n")
        f.write(f"# Total faces: {len(faces)}\n")
        f.write(f"# All normals point outward from sphere center\n\n")
        
        # Write vertices
        f.write("# Vertices (vertices)\n")
        for x, y, z in vertices:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        
        f.write("\n# Texture coordinates (texture coordinates)\n")
        for u, v in texture_coords:
            f.write(f"vt {u:.6f} {v:.6f}\n")
        
        f.write("\n# Normals (normals)\n")
        for nx, ny, nz in normals:
            f.write(f"vn {nx:.6f} {ny:.6f} {nz:.6f}\n")
        
        f.write("\n# Faces (faces)\n")
        for face in faces:
            # face format: (v1, vt1, vn1, v2, vt2, vn2, v3, vt3, vn3)
            f.write(f"f {face[0]}/{face[1]}/{face[2]} {face[3]}/{face[4]}/{face[5]} {face[6]}/{face[7]}/{face[8]}\n")

def main():
    print("Generating high-resolution sphere...")
    print("Resolution: 64 longitude divisions × 32 latitude divisions")
    print("This is 16x higher resolution than the original sphere")
    
    # Generate the sphere data
    vertices, texture_coords, normals, faces = generate_high_res_sphere(
        radius=2.0,
        longitude_divisions=64,  # 16 * 4 = 64
        latitude_divisions=32    # 8 * 4 = 32
    )
    
    # Write to file
    output_file = "assets/high_res_sphere.obj"
    write_obj_file(output_file, vertices, texture_coords, normals, faces)
    
    print(f"Generated {len(vertices)} vertices")
    print(f"Generated {len(faces)} faces")
    print(f"Output written to: {output_file}")

if __name__ == "__main__":
    main() 