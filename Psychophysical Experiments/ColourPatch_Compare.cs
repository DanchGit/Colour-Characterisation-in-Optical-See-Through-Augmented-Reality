using UnityEngine;
using UnityEngine.UI;
using TMPro;
using System.Collections.Generic;
using System.IO;
using UnityEngine.SceneManagement;
using System.Linq;


public class ColourPatch_Compare : MonoBehaviour
{
    public Renderer leftPatch;
    public Renderer middlePatch;
    public Renderer rightPatch;
    public TMP_Text colorNumberText;
    public TMP_Text colorNumberText2;
    public int colorNumber;
    public string sceneName;
    public AudioSource audioSource;
    public GameObject buffer;
    public GameObject setup;


    // Base Color Lists
    private List<(Color color, int number, string listName)> baseColors4 = new List<(Color, int, string)>
    {
        (new Color(0.493f, 0.214f, 0.132f), 1, "WhiteWall_Q2"),
        (new Color(0.803f, 0.420f, 0.293f), 2, "WhiteWall_Q2"),
        (new Color(0.454f, 0.332f, 0.390f), 3, "WhiteWall_Q2"),
        (new Color(0.391f, 0.286f, 0.107f), 4, "WhiteWall_Q2"),
        (new Color(0.533f, 0.322f, 0.419f), 5, "WhiteWall_Q2"),
        (new Color(0.523f, 0.542f, 0.396f), 6, "WhiteWall_Q2"),
        (new Color(0.904f, 0.349f, 0.030f), 7, "WhiteWall_Q2"),
        (new Color(0.364f, 0.225f, 0.414f), 8, "WhiteWall_Q2"),
        (new Color(0.764f, 0.186f, 0.205f), 9, "WhiteWall_Q2"),
        (new Color(0.389f, 0.095f, 0.229f), 10, "WhiteWall_Q2"),
        (new Color(0.668f, 0.539f, 0.074f), 11, "WhiteWall_Q2"),
        (new Color(0.874f, 0.428f, 0.000f), 12, "WhiteWall_Q2"),
        (new Color(0.270f, 0.105f, 0.394f), 13, "WhiteWall_Q2"),
        (new Color(0.433f, 0.440f, 0.136f), 14, "WhiteWall_Q2"),
        (new Color(0.692f, 0.000f, 0.103f), 15, "WhiteWall_Q2"),
        (new Color(0.960f, 0.580f, 0.000f), 16, "WhiteWall_Q2"),
        (new Color(0.697f, 0.123f, 0.331f), 17, "WhiteWall_Q2"),
        (new Color(0.261f, 0.349f, 0.384f), 18, "WhiteWall_Q2"),
        (new Color(1.000f, 0.750f, 0.596f), 19, "WhiteWall_Q2"),
        (new Color(0.877f, 0.611f, 0.507f), 20, "WhiteWall_Q2"),
        (new Color(0.681f, 0.454f, 0.390f), 21, "WhiteWall_Q2"),
        (new Color(0.523f, 0.333f, 0.267f), 22, "WhiteWall_Q2"),
        (new Color(0.382f, 0.216f, 0.173f), 23, "WhiteWall_Q2"),
        (new Color(0.295f, 0.121f, 0.117f), 24, "WhiteWall_Q2")
    };

    private List<(Color color, int number, string listName)> baseColors2 = new List<(Color, int, string)>
    {
        (new Color(0.470f, 0.352f, 0.282f), 1, "WhiteWall_Poly"),
        (new Color(0.744f, 0.691f, 0.520f), 2, "WhiteWall_Poly"),
        (new Color(0.407f, 0.397f, 0.692f), 3, "WhiteWall_Poly"),
        (new Color(0.307f, 0.413f, 0.247f), 4, "WhiteWall_Poly"),
        (new Color(0.491f, 0.419f, 0.732f), 5, "WhiteWall_Poly"),
        (new Color(0.596f, 0.633f, 0.678f), 6, "WhiteWall_Poly"),
        (new Color(0.620f, 0.580f, 0.219f), 7, "WhiteWall_Poly"),
        (new Color(0.164f, 0.221f, 0.746f), 8, "WhiteWall_Poly"),
        (new Color(0.825f, 0.421f, 0.354f), 9, "WhiteWall_Poly"),
        (new Color(0.269f, 0.197f, 0.442f), 10, "WhiteWall_Poly"),
        (new Color(0.351f, 0.787f, 0.256f), 11, "WhiteWall_Poly"),
        (new Color(0.480f, 0.685f, 0.226f), 12, "WhiteWall_Poly"),
        (new Color(0.000f, 0.099f, 0.710f), 13, "WhiteWall_Poly"),
        (new Color(0.351f, 0.607f, 0.293f), 14, "WhiteWall_Poly"),
        (new Color(0.816f, 0.223f, 0.179f), 15, "WhiteWall_Poly"),
        (new Color(0.231f, 0.872f, 0.210f), 16, "WhiteWall_Poly"),
        (new Color(0.743f, 0.359f, 0.513f), 17, "WhiteWall_Poly"),
        (new Color(0.259f, 0.327f, 0.693f), 18, "WhiteWall_Poly"),
        (new Color(0.970f, 1.000f, 0.990f), 19, "WhiteWall_Poly"),
        (new Color(0.849f, 0.839f, 0.848f), 20, "WhiteWall_Poly"),
        (new Color(0.679f, 0.647f, 0.667f), 21, "WhiteWall_Poly"),
        (new Color(0.508f, 0.482f, 0.508f), 22, "WhiteWall_Poly"),
        (new Color(0.308f, 0.314f, 0.354f), 23, "WhiteWall_Poly"),
        (new Color(0.190f, 0.217f, 0.252f), 24, "WhiteWall_Poly")
    };



    private List<(Color color, int number, string listName)> baseColors3 = new List<(Color, int, string)>
    {
        (new Color(0.164f, 0.161f, 0.083f), 1, "Painting"),
        (new Color(0.368f, 0.293f, 0.186f), 2, "Painting"),
        (new Color(0.117f, 0.227f, 0.252f), 3, "Painting"),
        (new Color(0.078f, 0.199f, 0.069f), 4, "Painting"),
        (new Color(0.171f, 0.225f, 0.275f), 5, "Painting"),
        (new Color(0.071f, 0.349f, 0.261f), 6, "Painting"),
        (new Color(0.425f, 0.259f, 0.054f), 7, "Painting"),
        (new Color(0.077f, 0.159f, 0.269f), 8, "Painting"),
        (new Color(0.377f, 0.175f, 0.131f), 9, "Painting"),
        (new Color(0.125f, 0.070f, 0.142f), 10, "Painting"),
        (new Color(0.210f, 0.353f, 0.061f), 11, "Painting"),
        (new Color(0.400f, 0.300f, 0.045f), 12, "Painting"),
        (new Color(0.000f, 0.067f, 0.252f), 13, "Painting"),
        (new Color(0.000f, 0.288f, 0.089f), 14, "Painting"),
        (new Color(0.350f, 0.063f, 0.069f), 15, "Painting"),
        (new Color(0.417f, 0.398f, 0.000f), 16, "Painting"),
        (new Color(0.347f, 0.133f, 0.205f), 17, "Painting"),
        (new Color(0.000f, 0.231f, 0.246f), 18, "Painting"),
        (new Color(0.446f, 0.513f, 0.409f), 19, "Painting"),
        (new Color(0.368f, 0.414f, 0.342f), 20, "Painting"),
        (new Color(0.275f, 0.304f, 0.257f), 21, "Painting"),
        (new Color(0.157f, 0.230f, 0.168f), 22, "Painting"),
        (new Color(0.100f, 0.154f, 0.104f), 23, "Painting"),
        (new Color(0.019f, 0.082f, 0.072f), 24, "Painting")
    };

    private List<(Color color, int number, string listName)> baseColors1 = new List<(Color, int, string)>
    {
        (new Color(0.157f, 0.078f, 0.059f), 1, "Physical_CC24"),
        (new Color(0.443f, 0.251f, 0.200f), 2, "Physical_CC24"),
        (new Color(0.094f, 0.161f, 0.271f), 3, "Physical_CC24"),
        (new Color(0.075f, 0.118f, 0.047f), 4, "Physical_CC24"),
        (new Color(0.157f, 0.157f, 0.310f), 5, "Physical_CC24"),
        (new Color(0.102f, 0.365f, 0.298f), 6, "Physical_CC24"),
        (new Color(0.600f, 0.180f, 0.043f), 7, "Physical_CC24"),
        (new Color(0.055f, 0.086f, 0.302f), 8, "Physical_CC24"),
        (new Color(0.439f, 0.086f, 0.118f), 9, "Physical_CC24"),
        (new Color(0.086f, 0.043f, 0.118f), 10, "Physical_CC24"),
        (new Color(0.255f, 0.357f, 0.055f), 11, "Physical_CC24"),
        (new Color(0.545f, 0.251f, 0.039f), 12, "Physical_CC24"),
        (new Color(0.031f, 0.043f, 0.278f), 13, "Physical_CC24"),
        (new Color(0.067f, 0.255f, 0.075f), 14, "Physical_CC24"),
        (new Color(0.357f, 0.031f, 0.047f), 15, "Physical_CC24"),
        (new Color(0.620f, 0.416f, 0.024f), 16, "Physical_CC24"),
        (new Color(0.341f, 0.067f, 0.204f), 17, "Physical_CC24"),
        (new Color(0.000f, 0.173f, 0.271f), 18, "Physical_CC24"),
        (new Color(0.722f, 0.710f, 0.675f), 19, "Physical_CC24"),
        (new Color(0.471f, 0.467f, 0.475f), 20, "Physical_CC24"),
        (new Color(0.278f, 0.278f, 0.286f), 21, "Physical_CC24"),
        (new Color(0.161f, 0.161f, 0.165f), 22, "Physical_CC24"),
        (new Color(0.078f, 0.078f, 0.082f), 23, "Physical_CC24"),
        (new Color(0.047f, 0.047f, 0.047f), 24, "Physical_CC24")
    };

    private int currentTrialIndex = 0;
    private List<(Color referenceColor, Color comparisonColor, int colorNumber, bool isRepeat, string comparisonString, string list1name, string list2name)> allComparisons;
    private Color comparisonColor;
    private int referenceSide;
    private int lastLoggedColorNumber = -1;
    private string logFilePath;
    //private float timer = 8f;  // Timer to count down from 5 seconds
    //private List<(Color referenceColor, Color comparisonColor, int colorNumber, bool isRepeat, string comparisonString, string list1name, string list2name)> unansweredTrials = new List<(Color, Color, int, bool, string, string, string)>();
    private List<(Color referenceColor, Color comparisonColor, int colorNumber, bool isRepeat, string comparisonString, string list1name, string list2name)> repeatComparisons = new();
    //private bool unansweredTrialsAdded = false;

    private void Start()
    {
        string savedText = PlayerPrefs.GetString("savedText", "defaultGUID");
        logFilePath = Path.Combine(Application.persistentDataPath, $"ColorSelectionLog_{savedText}.txt");

        // Build all comparisons into a single list (24 comparisons)
        allComparisons = new List<(Color referenceColor, Color comparisonColor, int colorNumber, bool isRepeat, string comparisonString, string list1name, string list2name)>();

        BuildColorShuffledComparisons();
        AddRandomRepeats();
        LogAllComparisons();
        BeginNextTrial();
    }


    private void BuildColorShuffledComparisons()
    {
        var rng = new System.Random();
        allComparisons = new List<(Color, Color, int, bool, string, string, string)>();

        var comparisons = new List<(List<(Color, int, string)>, List<(Color, int, string)>, string)>
    {
        (baseColors1, baseColors2, "Physical_CC24 vs WhiteWall_Poly"),
        (baseColors1, baseColors3, "Physical_CC24 vs Painting"),
        (baseColors1, baseColors4, "Physical_CC24 vs WhiteWall_Q2"),
        (baseColors2, baseColors3, "WhiteWall_Poly vs Painting"),
        (baseColors2, baseColors4, "WhiteWall_Poly vs WhiteWall_Q2"),
        (baseColors3, baseColors4, "Painting vs WhiteWall_Q2")
    };

        int colorCount = 24;
        int blocks = 6;

        // For each color, shuffle the 6 comparisons and store the order
        var perColorComparisonOrder = new List<List<int>>();
        for (int colorIndex = 0; colorIndex < colorCount; colorIndex++)
        {
            var order = Enumerable.Range(0, 6).OrderBy(x => rng.Next()).ToList();  // indices of comparisons
            perColorComparisonOrder.Add(order);
        }

        for (int block = 0; block < blocks; block++)
        {
            for (int colorIndex = 0; colorIndex < colorCount; colorIndex++)
            {
                int compIndex = perColorComparisonOrder[colorIndex][block];
                var (list1, list2, compName) = comparisons[compIndex];

                (Color color1, int number1, string listName1) = list1[colorIndex];
                (Color color2, int number2, string listName2) = list2[colorIndex];

                allComparisons.Add((
                    color1,
                    color2,
                    number1,  
                    false,
                    compName,
                    listName1,
                    listName2
                ));
            }
        }
    }



    private void AddComparisons(List<(Color, int, string)> list1, List<(Color, int, string)> list2, string comparisonString)
    {
        for (int i = 0; i < list1.Count; i++)  // Use list1.Count, not baseColors1.Count
        {
            int colorNumber = list1[i].Item2;  // colorNumber from list1
            string list1name = list1[i].Item3;
            string list2name = list2[i].Item3;
            allComparisons.Add((list1[i].Item1, list2[i].Item1, colorNumber, false, comparisonString, list1name, list2name));  
        }
    }


    private void LogAllComparisons()
    {
        Debug.Log("Logging all comparisons:");
        for (int i = 0; i < allComparisons.Count; i++)
        {
            var comparison = allComparisons[i];
            string logLine = $"Comparison {i + 1}: " +
                             $"Reference Color: {comparison.referenceColor} vs Comparison Color: {comparison.comparisonColor}";

            if (comparison.isRepeat)
            {
                logLine += $" (Repeat of {i - allComparisons.Count})";  
            }

            Debug.Log(logLine);
        }
    }
    private void BeginNextTrial()
    {
        
        audioSource.Play();
        setup.SetActive(false);
        buffer.SetActive(true);  
        Invoke("HideBuffer", 3f); 
        // Check if completed all comparisons (24 comparisons)
        if (currentTrialIndex >= allComparisons.Count)
        {
            Debug.Log("All comparisons complete.");
            SceneManager.LoadScene(sceneName); 
            return;
        }

        // Get the reference and comparison colors for this trial
        var (referenceColor, comparisonColor, colorNumber, isRepeat, comparisonString, list1name, list2name) = allComparisons[currentTrialIndex];


        // Log the last color number for reference
        lastLoggedColorNumber = colorNumber; 
        colorNumberText.text = "Color Number: " + colorNumber;
        colorNumberText2.text = "" + colorNumber;

        // Set the color patches for comparison
        SetPatchColors(referenceColor, comparisonColor);
        currentTrialIndex++;

        Debug.Log($"Trial {currentTrialIndex} started.");
    }

