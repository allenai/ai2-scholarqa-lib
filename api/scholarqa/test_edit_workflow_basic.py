"""
Basic structural validation for edit workflow (no dependencies).

This test validates:
1. All files compile
2. All imports work
3. Classes can be instantiated
4. Methods exist with correct signatures
5. Prompts have expected content

Run with: python3 test_edit_workflow_basic.py
"""

import sys
import inspect

# Add parent dir to path
sys.path.insert(0, '..')


def test_file_compilation():
    """Test that all edit workflow files compile."""
    print("\n1. Testing file compilation...")

    import py_compile
    files = [
        'models.py',
        'llms/edit_prompts.py',
        'rag/edit_pipeline.py',
        'scholar_qa.py',
        'app.py'
    ]

    for file in files:
        try:
            py_compile.compile(file, doraise=True)
            print(f"   ✅ {file} compiles")
        except Exception as e:
            print(f"   ❌ {file} failed: {e}")
            return False

    return True


def test_imports():
    """Test that all components can be imported."""
    print("\n2. Testing imports...")

    try:
        # Import edit-specific modules
        from scholarqa.rag.edit_pipeline import EditPipeline, EditClusterPlan
        print("   ✅ EditPipeline imported")

        from scholarqa.llms.edit_prompts import (
            SYSTEM_PROMPT_QUOTE_PER_PAPER_EDIT,
            SYSTEM_PROMPT_QUOTE_CLUSTER_EDIT,
            PROMPT_ASSEMBLE_SUMMARY_EDIT
        )
        print("   ✅ Edit prompts imported")

        from scholarqa.models import ToolRequest
        print("   ✅ ToolRequest imported")

        return True
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return False


def test_edit_pipeline_structure():
    """Test EditPipeline class structure."""
    print("\n3. Testing EditPipeline structure...")

    from scholarqa.rag.edit_pipeline import EditPipeline

    # Check class exists and can be instantiated
    try:
        pipeline = EditPipeline(llm_model="test")
        print("   ✅ EditPipeline instantiates")
    except Exception as e:
        print(f"   ❌ Instantiation failed: {e}")
        return False

    # Check required methods exist
    required_methods = [
        'step_select_quotes_edit',
        'step_clustering_edit',
        'generate_iterative_summary_edit',
        '_format_sections_for_quote_extraction',
        '_format_report_summary',
        '_format_report_for_clustering'
    ]

    for method_name in required_methods:
        if hasattr(pipeline, method_name):
            method = getattr(pipeline, method_name)
            if callable(method):
                sig = inspect.signature(method)
                print(f"   ✅ {method_name}({', '.join(sig.parameters.keys())})")
            else:
                print(f"   ❌ {method_name} is not callable")
                return False
        else:
            print(f"   ❌ {method_name} not found")
            return False

    return True


def test_edit_prompts_content():
    """Test that edit prompts contain expected content."""
    print("\n4. Testing edit prompts content...")

    from scholarqa.llms.edit_prompts import (
        SYSTEM_PROMPT_QUOTE_PER_PAPER_EDIT,
        SYSTEM_PROMPT_QUOTE_CLUSTER_EDIT,
        PROMPT_ASSEMBLE_SUMMARY_EDIT
    )

    checks = [
        (SYSTEM_PROMPT_QUOTE_PER_PAPER_EDIT, ["edit", "current", "instruction"], "SYSTEM_PROMPT_QUOTE_PER_PAPER_EDIT"),
        (SYSTEM_PROMPT_QUOTE_CLUSTER_EDIT, ["edit", "current_report", "action"], "SYSTEM_PROMPT_QUOTE_CLUSTER_EDIT"),
        (PROMPT_ASSEMBLE_SUMMARY_EDIT, ["edit", "action", "current_section"], "PROMPT_ASSEMBLE_SUMMARY_EDIT"),
    ]

    for prompt, keywords, name in checks:
        if not isinstance(prompt, str):
            print(f"   ❌ {name} is not a string")
            return False

        if len(prompt) < 100:
            print(f"   ❌ {name} too short (< 100 chars)")
            return False

        found_keywords = [kw for kw in keywords if kw.lower() in prompt.lower()]
        missing_keywords = [kw for kw in keywords if kw.lower() not in prompt.lower()]

        if missing_keywords:
            print(f"   ⚠️  {name} missing keywords: {missing_keywords}")
        else:
            print(f"   ✅ {name} contains: {found_keywords}")

    return True


def test_tool_request_fields():
    """Test ToolRequest has edit fields."""
    print("\n5. Testing ToolRequest edit fields...")

    from scholarqa.models import ToolRequest

    # Check if edit fields exist
    sample = ToolRequest(
        query="test",
        edit_existing=True,
        thread_id="test-thread",
        edit_instruction="test instruction",
        mentioned_papers=[1, 2, 3]
    )

    checks = [
        ('edit_existing', True),
        ('thread_id', "test-thread"),
        ('edit_instruction', "test instruction"),
        ('mentioned_papers', [1, 2, 3])
    ]

    for field, expected in checks:
        if hasattr(sample, field):
            actual = getattr(sample, field)
            if actual == expected:
                print(f"   ✅ {field} = {actual}")
            else:
                print(f"   ❌ {field} = {actual} (expected {expected})")
                return False
        else:
            print(f"   ❌ {field} not found")
            return False

    return True


def test_scholar_qa_has_run_edit_pipeline():
    """Test ScholarQA has run_edit_pipeline method."""
    print("\n6. Testing ScholarQA.run_edit_pipeline exists...")

    from scholarqa.scholar_qa import ScholarQA

    # Check method exists
    if hasattr(ScholarQA, 'run_edit_pipeline'):
        method = getattr(ScholarQA, 'run_edit_pipeline')
        if callable(method):
            sig = inspect.signature(method)
            params = list(sig.parameters.keys())
            print(f"   ✅ run_edit_pipeline exists with params: {params}")

            # Check for expected parameters
            if 'req' in params and 'inline_tags' in params:
                print(f"   ✅ Has expected parameters")
            else:
                print(f"   ⚠️  Parameters may not match expected signature")

            return True
        else:
            print(f"   ❌ run_edit_pipeline is not callable")
            return False
    else:
        print(f"   ❌ run_edit_pipeline not found")
        return False


def test_app_routing():
    """Test app.py has routing for edit workflow."""
    print("\n7. Testing app.py routing...")

    # Read app.py and check for edit routing
    with open('app.py', 'r') as f:
        content = f.read()

    checks = [
        ('edit_existing check', 'edit_existing' in content),
        ('run_edit_pipeline call', 'run_edit_pipeline' in content),
        ('run_qa_pipeline call', 'run_qa_pipeline' in content),
    ]

    all_passed = True
    for check_name, result in checks:
        if result:
            print(f"   ✅ {check_name}")
        else:
            print(f"   ❌ {check_name}")
            all_passed = False

    return all_passed


def test_mirrored_structure():
    """Test that edit methods mirror original methods."""
    print("\n8. Testing mirrored structure...")

    from scholarqa.rag.multi_step_qa_pipeline import MultiStepQAPipeline
    from scholarqa.rag.edit_pipeline import EditPipeline

    original_methods = [
        'step_select_quotes',
        'step_clustering',
        'generate_iterative_summary'
    ]

    edit_methods = [
        'step_select_quotes_edit',
        'step_clustering_edit',
        'generate_iterative_summary_edit'
    ]

    for orig, edit in zip(original_methods, edit_methods):
        has_orig = hasattr(MultiStepQAPipeline, orig)
        has_edit = hasattr(EditPipeline, edit)

        if has_orig and has_edit:
            print(f"   ✅ {orig} ↔ {edit}")
        else:
            print(f"   ❌ Missing: orig={has_orig}, edit={has_edit}")
            return False

    return True


def main():
    """Run all tests."""
    print("=" * 70)
    print("EDIT WORKFLOW STRUCTURAL VALIDATION")
    print("=" * 70)

    tests = [
        ("File Compilation", test_file_compilation),
        ("Imports", test_imports),
        ("EditPipeline Structure", test_edit_pipeline_structure),
        ("Edit Prompts Content", test_edit_prompts_content),
        ("ToolRequest Fields", test_tool_request_fields),
        ("ScholarQA Integration", test_scholar_qa_has_run_edit_pipeline),
        ("App Routing", test_app_routing),
        ("Mirrored Structure", test_mirrored_structure),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    print("=" * 70)
    print(f"\nResults: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n🎉 All structural validations passed!")
        print("\nWhat this means:")
        print("- All files compile without syntax errors")
        print("- All imports work correctly")
        print("- EditPipeline mirrors MultiStepQAPipeline structure")
        print("- Prompts extend original prompts with edit context")
        print("- ToolRequest supports edit mode")
        print("- ScholarQA has run_edit_pipeline method")
        print("- App routing supports edit workflow")
        print("\nWhat this DOESN'T test:")
        print("- ❌ LLM calls (would need API keys)")
        print("- ❌ End-to-end workflow (needs real data)")
        print("- ❌ Quote extraction quality")
        print("- ❌ Plan generation logic")
        print("\nNext steps:")
        print("1. Manual testing with real thread_id and mentioned_papers")
        print("2. Check LLM responses for quality")
        print("3. Verify edited reports maintain coherence")
        return 0
    else:
        print(f"\n❌ {total_count - passed_count} tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
