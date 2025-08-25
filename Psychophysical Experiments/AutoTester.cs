using UnityEngine;
using System.Collections;

public class AutoTester : MonoBehaviour
{
    public enum SelectionMode
    {
        AlwaysLeft,
        AlwaysRight,
        Random
    }

    [Header("Auto Test Settings")]
    [Tooltip("Enable to run autotester automatically on Start")]
    public bool enableAutoTest = false;

    [Tooltip("How many test iterations to run")]
    public int testIterations = 5;

    [Tooltip("Choose the selection mode for auto testing")]
    public SelectionMode selectionMode = SelectionMode.AlwaysLeft;

    [Header("References")]
    [Tooltip("Assign your ColourPatch_Compare script here")]
    public ColourPatch_Sat compareScript;

    void Start()
    {

        if (enableAutoTest)
        {
            if (compareScript == null)
            {
                Debug.LogError("AutoTester: compareScript reference is missing!");
                return;
            }
            StartCoroutine(RunAutoTest());
        }
    }

    IEnumerator RunAutoTest()
    {
        Debug.Log("AutoTester: Starting automated test...");

        for (int i = 0; i < testIterations; i++)
        {
            bool selectLeft;

            switch (selectionMode)
            {
                case SelectionMode.AlwaysLeft:
                    selectLeft = true;
                    break;
                case SelectionMode.AlwaysRight:
                    selectLeft = false;
                    break;
                case SelectionMode.Random:
                    selectLeft = (Random.value > 0.5f);
                    break;
                default:
                    selectLeft = true;
                    break;
            }

            Debug.Log($"AutoTester: Iteration {i + 1}, Selection Mode: {selectionMode}, Selecting {(selectLeft ? "LEFT" : "RIGHT")}");


            if (selectLeft)
            {
                compareScript.SelectLeft();  
            }
            else
            {
                compareScript.SelectRight(); 
            }

            yield return new WaitForSeconds(1f);
        }

        Debug.Log("AutoTester: Automated test completed.");
    }
}
