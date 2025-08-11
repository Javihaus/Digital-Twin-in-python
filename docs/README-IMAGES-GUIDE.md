# README Image Guide

## Image Storage Locations
- **Primary**: `docs/images/` (recommended)
- **Alternative**: `assets/`
- **GitHub specific**: `.github/assets/`

## Image Formatting Options

### 1. Centered with Fixed Width
```markdown
<div align="center">
  <img src="docs/images/your-image.png" alt="Description" width="600"/>
</div>
```

### 2. Centered with Percentage Width (Responsive)
```markdown
<div align="center">
  <img src="docs/images/your-image.png" alt="Description" width="80%"/>
</div>
```

### 3. Centered with Max Width (Best for large images)
```markdown
<div align="center">
  <img src="docs/images/your-image.png" alt="Description" style="max-width: 700px; width: 100%;"/>
</div>
```

### 4. With Caption
```markdown
<div align="center">
  <img src="docs/images/your-image.png" alt="Description" width="600"/>
  <br>
  <em>Your image caption here</em>
</div>
```

### 5. Side-by-side Images
```markdown
<div align="center">
  <img src="docs/images/image1.png" alt="Description 1" width="45%"/>
  <img src="docs/images/image2.png" alt="Description 2" width="45%"/>
</div>
```

## Supported Image Formats
- PNG (recommended for diagrams/screenshots)
- JPG/JPEG (good for photos)
- SVG (best for vector graphics, scalable)
- GIF (for animations)

## Image Size Recommendations
- **Header images**: 700-800px width
- **Diagrams**: 600-700px width
- **Screenshots**: 500-600px width
- **Icons/small images**: 100-200px width

## Best Practices
1. Use descriptive alt text for accessibility
2. Keep file sizes under 1MB when possible
3. Use PNG for technical diagrams
4. Name files descriptively (e.g., `architecture-overview.png`)
5. Add captions for complex diagrams