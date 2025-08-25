using System.Collections;
using System.IO;
using UnityEngine;
using UnityEngine.UI; 
using UnityEngine.Audio; 
using MixedReality.Toolkit.UX;
using UnityEngine.SceneManagement;




public class JND : MonoBehaviour
{
    public Renderer targetRenderer;
    public Renderer secondRenderer;
    private static int switchCount = 0;
    private const int totalSteps = 15;
    public MixedReality.Toolkit.UX.Slider sliderComponent;
    private Color staticColor = new Color(1f, 0f, 0f);
    private string logFilePath;
    public string sceneName;



    void Start()
    {
        string shortGuid = PlayerPrefs.GetString("savedText", null);
        logFilePath = Path.Combine(Application.persistentDataPath, $"JND_{shortGuid}.txt");
    }

    public void UpdateColor(SliderEventData sliderEventData)
    {
        float jndSteps = sliderEventData.NewValue;
        Color newColor = Color.black;
        Vector3 labRef = ColorConverter.RGBToLAB(staticColor);
        Vector3 labNew = labRef;

        switch (switchCount % 5)
        {
            case 0: labNew.y -= jndSteps * 5f; break; // Change a* for red
            case 1: labNew.y += jndSteps * 5f; break; // Change a* for green
            case 2: labNew.z += jndSteps * 5f; break; // Change b* for blue
            case 3: labNew.z -= jndSteps * 5f; break; // Change b* for yellow
            case 4: labNew.x -= jndSteps * 5f; break; // Change L* for grayscale
            default: return;
        }

        newColor = ColorConverter.LABToRGB(labNew);

        if (targetRenderer != null)
            targetRenderer.material.color = newColor;
    }


    public void LoadNextScene()
    {
        if (!string.IsNullOrEmpty(sceneName))
        {
            SceneManager.LoadScene(sceneName);
        }
    }
    public void CycleColor()
    {

        if (targetRenderer != null && secondRenderer != null)
        {
            Color col1 = targetRenderer.material.color;
            Color col2 = secondRenderer.material.color;

            string logLine = $"Step {switchCount + 1} - Selected RGB: ({col1.r:F3}, {col1.g:F3}, {col1.b:F3}) | Reference RGB: ({col2.r:F3}, {col2.g:F3}, {col2.b:F3})";
            File.AppendAllText(Path.Combine(Application.persistentDataPath, logFilePath), logLine + "\n");
            Vector3 selectedLAB = ColorConverter.RGBToLAB(col1);
            Vector3 referenceLAB = ColorConverter.RGBToLAB(col2);
            string logLine2 = $"Step {switchCount + 1} - Selected LAB: ({selectedLAB.x:F1}, {selectedLAB.y:F1}, {selectedLAB.z:F1}) | Reference LAB: ({referenceLAB.x:F1}, {referenceLAB.y:F1}, {referenceLAB.z:F1})";
            File.AppendAllText(Path.Combine(Application.persistentDataPath, logFilePath), logLine2 + "\n");

        }
        switchCount++;
        if (switchCount >= totalSteps)
        {
            LoadNextScene();
        }
        if (sliderComponent != null)
            sliderComponent.Value = 0f;
        switch (switchCount % 5)
        {
            case 0:
                if (targetRenderer != null)
                    targetRenderer.material.color = new Color(1f, 0f, 0f);
                staticColor = new Color(1f, 0f, 0f);  // Red
                if (secondRenderer != null)
                    secondRenderer.material.color = staticColor;
                break;
            case 1:
                if (targetRenderer != null)
                    targetRenderer.material.color = new Color(0f, 1f, 0f);
                staticColor = new Color(0f, 1f, 0f);  // Green
                if (secondRenderer != null)
                    secondRenderer.material.color = staticColor;
                break;
            case 2:
                if (targetRenderer != null)
                    targetRenderer.material.color = new Color(0f, 0f, 1f);
                staticColor = new Color(0f, 0f, 1f);  // Blue
                if (secondRenderer != null)
                    secondRenderer.material.color = staticColor;
                break;
            case 3:
                if (targetRenderer != null)
                    targetRenderer.material.color = new Color(1f, 1f, 0f);
                staticColor = new Color(1f, 1f, 0f);  // Yellow
                if (secondRenderer != null)
                    secondRenderer.material.color = staticColor;
                break;
            case 4:
                if (targetRenderer != null)
                    targetRenderer.material.color = new Color(1f, 1f, 1f);
                staticColor = new Color(1f, 1f, 1f);  // White
                if (secondRenderer != null)
                    secondRenderer.material.color = staticColor;
                break;
        }


    }

    // LAB
    public static class ColorConverter
    {
        // D65 standard reference white
        private static readonly Vector3 D65 = new Vector3(95.047f, 100f, 108.883f);

        public static Vector3 RGBToLAB(Color color)
        {
            // Convert to XYZ first
            float r = PivotRGB(color.r);
            float g = PivotRGB(color.g);
            float b = PivotRGB(color.b);

            // sRGB  XYZ matrix (D65)
            float x = r * 0.4124f + g * 0.3576f + b * 0.1805f;
            float y = r * 0.2126f + g * 0.7152f + b * 0.0722f;
            float z = r * 0.0193f + g * 0.1192f + b * 0.9505f;

            // Normalize
            x /= D65.x;
            y /= D65.y;
            z /= D65.z;

            x = PivotXYZ(x);
            y = PivotXYZ(y);
            z = PivotXYZ(z);

            float L = 116f * y - 16f;
            float a = 500f * (x - y);
            float bVal = 200f * (y - z);

            return new Vector3(L, a, bVal);
        }

        public static Color LABToRGB(Vector3 lab)
        {
            float y = (lab.x + 16f) / 116f;
            float x = lab.y / 500f + y;
            float z = y - lab.z / 200f;

            x = InversePivotXYZ(x) * D65.x;
            y = InversePivotXYZ(y) * D65.y;
            z = InversePivotXYZ(z) * D65.z;

            // XYZ to Linear RGB
            float r = x * 3.2406f + y * -1.5372f + z * -0.4986f;
            float g = x * -0.9689f + y * 1.8758f + z * 0.0415f;
            float b = x * 0.0557f + y * -0.2040f + z * 1.0570f;

            // Gamma correction
            r = Mathf.Clamp01(InversePivotRGB(r));
            g = Mathf.Clamp01(InversePivotRGB(g));
            b = Mathf.Clamp01(InversePivotRGB(b));

            return new Color(r, g, b);
        }

        private static float PivotRGB(float c)
        {
            return (c > 0.04045f) ? Mathf.Pow((c + 0.055f) / 1.055f, 2.4f) : c / 12.92f;
        }

        private static float InversePivotRGB(float c)
        {
            return (c > 0.0031308f) ? 1.055f * Mathf.Pow(c, 1f / 2.4f) - 0.055f : 12.92f * c;
        }

        private static float PivotXYZ(float c)
        {
            return (c > 0.008856f) ? Mathf.Pow(c, 1f / 3f) : (7.787f * c + 16f / 116f);
        }

        private static float InversePivotXYZ(float c)
        {
            float c3 = Mathf.Pow(c, 3f);
            return (c3 > 0.008856f) ? c3 : (c - 16f / 116f) / 7.787f;
        }
    }

}